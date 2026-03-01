# 🚀 1.5B SLM Reasoning & Coding Pipeline

This project explores specialized fine-tuning and reasoning distillation for Small Language Models (SLMs) in the domain of Competitive Programming.

---

## 🏗️ Project Architecture: Models & Data

### **Core Model**
* **Base Model:** `DeepSeek-R1-Distill-Qwen-1.5B`
* **Architecture:** A distilled reasoning model optimized for Chain-of-Thought (CoT) processing.

### **Datasets**
1.  **rStar-Coder (Training):**
    * **Role:** Supervised Fine-Tuning (SFT).
    * **Feature:** Contains **Verified Chain-of-Thought (CoT)** reasoning traces. This is used to teach the model how to "think" before it "codes."
2.  **Open-R1 (Evaluation):**
    * **Role:** Performance Benchmarking.
    * **Source:** `open-r1/verifiable-coding-problems-python` (Decontaminated).

---

## 📊 Current Progress: Phase 0 (Baseline)

As of now, the following components have been established and executed:

### **1. The Test Suite (`test_suite_200_FINAL.jsonl`)**
To accurately measure progress, a stratified baseline test suite of **200 problems** was curated. This ensures the model is evaluated across a balanced spectrum of difficulty:
* **Easy:** 60 Problems (Basic logic & syntax)
* **Medium:** 80 Problems (Standard algorithms/DS)
* **Hard:** 60 Problems (Complex optimization & edge cases)

### **2. Inference Engine (`SLM_notebook_2.ipynb`)**
This notebook handles the zero-shot inference of the baseline model.
* **Function:** Processes the 200 problems in the test suite using Unsloth-optimized batched inference (Batch Size: 10).
* **Output:** Generates a file named `model_solutions_batched.jsonl`.
* **Content:** This output file captures the model's raw reasoning (`<think>`) and Python implementation (`<code>`).

---

## 🏁 Immediate Next Steps

1.  **Local Evaluation:** The `model_solutions_batched.jsonl` will be passed through a **Local Judge (Sandbox)** to produce baseline score.
2.  **Scoring:** The judge will execute the generated code against hidden test cases to produce the initial **Pass@1** baseline score.

## 📏 Performance Metrics & Evaluation Framework

To quantify the model's reasoning and coding proficiency, we utilize three primary metrics. These are calculated using a local Python Sandbox on Fedora to ensure execution safety and accuracy.

### **1. Pass@1 (Functional Correctness)**
This is the primary industry standard for coding models. 
* **Definition:** The percentage of problems where the model's first generated solution passes all hidden test cases.
* **Calculation:** $$\text{Pass@1} = \frac{\text{Total Solved Tasks}}{\text{Total Tasks (200)}}$$
* **Significance:** Measures the model's ability to produce bug-free code in a single attempt.

### **2. Reasoning Density (CoT Quality)**
Since we are using an R1-distilled model, we measure the "Reasoning-to-Code" ratio.
* **Definition:** The average number of tokens generated inside the `<think>` tags compared to the `<code>` tags.
* **Significance:** Helps identify if the model is "thinking" deeply enough for Hard problems or if it is skipping the logic phase (which often leads to `pass` placeholders).

### **3. Format Adherence Rate (FAR)**
A specialized metric for SLMs to ensure they follow structural constraints.
* **Definition:** The percentage of outputs that correctly include both opening and closing `<think>` and `<code>` tags.
* **Significance:** High FAR indicates that the 1.5B model is successfully steered by the System Prompt and hasn't "collapsed" during long inference chains.

### **4. Difficulty-Categorized Accuracy**
By utilizing our **60-80-60 stratified split**, we track performance across tiers:
* **Easy Pass Rate:** Baseline for syntax and basic logic.
* **Medium Pass Rate:** Measures algorithmic implementation (BFS, DFS, Greedy).
* **Hard Pass Rate:** Tests complex optimization and high-level reasoning.


  # Gdrive id for dataset and Model weights: 
  * To gain access: https://drive.google.com/drive/folders/1UcsTRmEJnTe0ogOkDZh0Oxj1DH5JyxYs?usp=drive_link
  * To download: https://drive.google.com/drive/folders/1UcsTRmEJnTe0ogOkDZh0Oxj1DH5JyxYs
