
# 🧠 Face Recognition on Low-Quality Images  
**AI Developer Intern Task – Jezt Technologies**  
**Author:** Vaikhashi A

---

## 📌 Project Overview

This project addresses real-world challenges in facial recognition using **low-quality images**. It employs a combination of a **pretrained InsightFace model** for robust baseline embeddings and a **fine-tuned ResNet18** classifier to improve performance.

The system performs:
- Face detection & embedding extraction (InsightFace)
- Baseline evaluation (ROC, Precision-Recall, Similarity Distribution)
- Model fine-tuning using a custom ResNet18 classifier
- Post-finetuning performance comparison

---

## 🔧 Setup Instructions

### 1. Clone & Environment Setup

```bash
git clone https://github.com/Vaikhashi/FaceRecognitionTask.git
cd FaceRecognitionTask
python -m venv venv
venv\Scripts\activate          # For Windows
pip install -r requirements.txt
````

---

### 2. Prepare the Dataset

* Download the dataset from the link provided by Jezt Technologies
* Place the dataset inside an `images/` folder like this:

```
images/
├── person1/
│   └── ananthu/
│       ├── High_quality/
│       └── Low_quality/
├── person2/
│   └── ...
```

---

### 3. Execute the Pipeline

```bash
# Split dataset into training and validation
python split_dataset.py

# Run baseline embedding extraction
python baseline.py

# Generate baseline performance plots
python baseline_evaluation.py

# Fine-tune the ResNet18 model on low-quality faces
python fine_tune.py

# Evaluate fine-tuned model and generate final plots
python evaluate_fine_tuned.py
```

---

## 📁 Project Files

| File                                | Description                                                 |
| ----------------------------------- | ----------------------------------------------------------- |
| `baseline.py`                       | Extracts embeddings using InsightFace                       |
| `baseline_evaluation.py`            | ROC, Precision-Recall, and Similarity plots (before tuning) |
| `fine_tune.py`                      | Fine-tunes a ResNet18 model on training set                 |
| `evaluate_fine_tuned.py`            | Evaluates model performance after tuning                    |
| `split_dataset.py`                  | Organizes dataset into `train/` and `val/` splits           |
| `AI_Developer_Report_Vaikhashi.pdf` | 📄 Final evaluation report with summary, results & plots    |


## 🙏 Acknowledgement

Thank you Jezt Technologies for the opportunity to showcase my skills on a practical AI problem.
This task allowed me to explore real-world challenges in face recognition and apply both pretrained models and deep learning techniques to address them.





