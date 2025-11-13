

# Deep-SVDD Flaky Test Log Anomaly Detector

*A neural network–based anomaly detection system for identifying flaky test behavior from raw CI logs.*

---

## 📌 Overview

Flaky tests are one of the most costly and time-consuming issues in large-scale CI/CD systems.
They introduce:

* False failures
* Unstable builds
* Increased triage cost
* Reduced developer trust in test results

This project builds a **neural network–driven anomaly detection engine** using **Deep SVDD** to analyze raw test logs, identify unstable execution patterns, and highlight *flaky-like* behavior without requiring test re-runs.



---

# 🧠 Project Architecture

```
                  +------------------------+
                  |    Raw Test Logs       |
                  |   (JUnit / pytest)     |
                  +-----------+------------+
                              |
                              v
                 +-------------------------+
                 |   Log Preprocessing     |
                 | tokenize + pad + vocab  |
                 +-------------+-----------+
                               |
                               v
                +--------------------------+
                |  LSTM Log Encoder (NN)   |
                |  → Embedding Vector      |
                +-------------+------------+
                              |
                              v
                +--------------------------+
                |     Deep SVDD Model      |
                |  Learn normal behavior   |
                +-------------+------------+
                              |
                              v
         +-------------------------------------------+
         |  CI Pipeline Integration (score_logs.py)  |
         |  anomaly_score + flaky_like prediction    |
         +-------------------------------------------+
```

---

# 🚀 Features

### 🔍 Neural Network–Powered Log Embeddings

* Custom **LSTM** encoder trained on thousands of log lines
* Captures sequential execution behavior
* Robust to noise, varying patterns, long logs

### 🔒 Deep SVDD-Based Anomaly Detection

* Learns a hypersphere of "normal" test behavior
* High anomaly scores → flaky-like behavior
* No need for labeled data（unsupervised）

### 🧪 CI Pipeline Integration

* One command to score any new test log
* Outputs:

  ```json
  {
    "log_file": "test_login.log",
    "anomaly_score": 0.83,
    "flagged_as_flaky_like": true
  }
  ```

### 📊 Evaluation Support (Optional)

* If labeled flaky test logs are available
* Computes ROC AUC / PR AUC / F1
* Visualizes anomaly score distribution

---

# 📁 Repository Structure

```
deep-svdd-flaky-log-detector/
├── README.md
├── requirements.txt
├── data/
│   ├── raw_logs/
│   ├── processed/
│   └── labels.csv          # (optional)
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   ├── models/
│   │   ├── encoder.py
│   │   └── deep_svdd.py
│   └── ci_demo/
│       ├── simulate_test_run.py
│       └── run_ci_pipeline.py
└── notebooks/
    ├── 01_explore_logs.ipynb
    └── 02_visualize_scores.ipynb
```

---

# ⚙️ Installation

```bash
git clone https://github.com/yourname/deep-svdd-flaky-log-detector
cd deep-svdd-flaky-log-detector

pip install -r requirements.txt
```

---

# 🧪 Dataset

This project can use:

### ✓ **IDoFT — International Dataset of Flaky Tests**

Used to simulate real flaky behavior patterns and log variability.

or

### ✓ Your own CI logs

* JUnit XML logs
* pytest output
* Application logs
* Integration test logs

---

# 🔧 Training

### 1. Put logs into:

```
data/raw_logs/*.log
```

### 2. Train encoder + Deep SVDD:

```bash
python -m src.train
```

This will:

* Build vocabulary
* Train LSTM encoder
* Extract embeddings
* Fit Deep SVDD
* Save model under `data/processed/`

---

# 📈 Evaluation (Optional)

```bash
python -m src.evaluate
```

Metrics:

* ROC AUC
* PR AUC
* Anomaly score histograms

> 📌 **TODO:** Add your results here
>
> * ROC AUC: **??%**
> * PR AUC: **??%**

---

# 🔌 CI Pipeline Simulation

Example:

```bash
python -m src.ci_demo.run_ci_pipeline --log data/raw_logs/sample.log
```

Example output:

```json
{
  "log_file": "sample.log",
  "anomaly_score": 0.91,
  "flagged_as_flaky_like": true
}
```

Integratable with:

* Jenkins
* GitHub Actions
* GitLab CI
* Azure DevOps

---

# 📊 Results

> 📌 TODO: Add your screenshots here


### 1. Anomaly Score Distribution Plot

* Normal vs flaky log score separation

### 2. Example CI Output

* JSON summary from pipeline

### 3. ROC / PR Curves

* If labels are available

---

# 🧠 Why Deep SVDD for Flaky Tests?

Flaky test detection usually requires:

* expensive reruns
* heuristics
* static analysis
* change analysis

**Deep SVDD avoids all these by analyzing behavior directly from logs.**


---




# 🧾 License

MIT

---

