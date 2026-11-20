"""Track A3: the System-One controller.

Arm A (default): typed heads on the coder's own last-layer, post-norm hidden state at the last prompt
token (the exact point where generation starts, so a phone can reuse the same prefill).
Outputs, all from one forward pass:
  pass_logit[a]  -> P(pass | arm a)            (Jev "Noul" per arm; Choice = argmax of utility)
  log_tokens[a]  -> expected tokens of arm a   (Score)
  topic_logit[f] -> soft topic families        (auxiliary; also feeds the pred_tags ladder rung)
Trained with proper scoring rules against soft targets (the coder's measured pass rates), then
temperature-scaled on a held-out split.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ features

@torch.no_grad()
def prefill_features(model_path: str, prompts: list[str], batch_size: int = 16, max_len: int = 4096,
                     device: str = "cuda", dtype=torch.bfloat16) -> np.ndarray:
    """Post-norm last-layer hidden state at the final prompt token, shape (N, d), float32."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
    feats = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len, add_special_tokens=False).to(device)
        h = model(**enc).last_hidden_state  # base model output is after the final norm
        feats.append(h[:, -1, :].float().cpu().numpy())  # left padding: last position is the last token
    return np.concatenate(feats)


# ------------------------------------------------------------------ heads

class ControllerHeads(nn.Module):
    def __init__(self, d_in: int, n_arms: int, n_topics: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.trunk = (nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout))
                      if hidden else nn.Identity())
        d = hidden or d_in
        self.pass_head = nn.Linear(d, n_arms)
        self.tok_head = nn.Linear(d, n_arms)
        self.topic_head = nn.Linear(d, n_topics)
        # temperatures set by calibrate(); 1.0 = uncalibrated
        self.register_buffer("t_pass", torch.ones(1))
        self.register_buffer("t_topic", torch.ones(1))
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sigma", torch.ones(d_in))

    def forward(self, x):
        z = self.trunk((x - self.mu) / self.sigma)
        return self.pass_head(z), self.tok_head(z), self.topic_head(z)

    @torch.no_grad()
    def predict(self, x: np.ndarray):
        self.eval()
        pl, lt, tl = self(torch.as_tensor(x, dtype=torch.float32))
        return (torch.sigmoid(pl / self.t_pass).numpy(), torch.expm1(lt).numpy(),
                torch.sigmoid(tl / self.t_topic).numpy())


def _loss(model, x, R, W, logT, topics, tmask, w_tok=0.1, w_topic=0.1):
    pl, lt, tl = model(x)
    # soft-target BCE = proper scoring rule for P(pass|arm); weighted by #samples behind each rate
    l_pass = (F.binary_cross_entropy_with_logits(pl, R, reduction="none") * W).sum() / W.sum()
    l_tok = F.mse_loss(lt, logT)
    l_top = ((F.binary_cross_entropy_with_logits(tl, topics, reduction="none").mean(1) * tmask).sum()
             / tmask.sum().clamp(min=1))
    return l_pass + w_tok * l_tok + w_topic * l_top, l_pass


def train_heads(X_tr, R_tr, N_tr, T_tr, Y_tr, X_va, R_va, N_va, T_va, Y_va, hidden=256, lr=1e-3,
                weight_decay=1e-2, epochs=200, batch_size=256, patience=15, seed=0) -> ControllerHeads:
    torch.manual_seed(seed)
    model = ControllerHeads(X_tr.shape[1], R_tr.shape[1], Y_tr.shape[1], hidden)
    model.mu.copy_(torch.as_tensor(X_tr.mean(0)))
    model.sigma.copy_(torch.as_tensor(X_tr.std(0) + 1e-6))
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    tr = [t(X_tr), t(R_tr), t(N_tr), t(np.log1p(T_tr)), t(Y_tr), t(Y_tr.sum(1) > 0)]
    va = [t(X_va), t(R_va), t(N_va), t(np.log1p(T_va)), t(Y_va), t(Y_va.sum(1) > 0)]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, bad = float("inf"), None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr[0]), generator=g)
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            loss, _ = _loss(model, *[v[b] for v in tr])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            _, vp = _loss(model, *va)
        if vp.item() < best - 1e-5:
            best, bad = vp.item(), 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def _fit_temperature(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None) -> float:
    logt = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logt], lr=0.1, max_iter=200)
    w = torch.ones_like(targets) if weights is None else weights

    def closure():
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(logits / logt.exp(), targets, reduction="none") * w).sum() / w.sum()
        loss.backward()
        return loss
    opt.step(closure)
    return float(logt.exp())


@torch.no_grad()
def _logits(model, X):
    model.eval()
    return model(torch.as_tensor(X, dtype=torch.float32))


def calibrate(model: ControllerHeads, X_va, R_va, N_va, Y_va) -> ControllerHeads:
    pl, _, tl = _logits(model, X_va)
    with torch.enable_grad():
        model.t_pass.fill_(_fit_temperature(pl, torch.as_tensor(R_va, dtype=torch.float32),
                                            torch.as_tensor(N_va, dtype=torch.float32)))
        m = Y_va.sum(1) > 0
        if m.any():
            model.t_topic.fill_(_fit_temperature(tl[torch.as_tensor(m)], torch.as_tensor(Y_va[m], dtype=torch.float32)))
    return model


def export_numpy(model: ControllerHeads, path: str, meta: dict) -> None:
    """Weights for the phone (pure numpy forward in phone/pipeline.py)."""
    sd = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(path, **sd, meta=np.array(__import__("json").dumps(meta)))


from .heads_np import numpy_forward  # noqa: E402,F401  (re-export; the phone imports heads_np directly)
