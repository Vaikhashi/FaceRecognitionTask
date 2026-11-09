# Face recognition on low-quality images

Face matching usually works fine on clear photos and falls apart on blurry, dark or tiny ones. This project looks at how much a model can recover on the low-quality ones.

I started with a pretrained InsightFace model to get face embeddings as a baseline, then fine-tuned a ResNet18 on the low-quality images and compared the two. I built it as a take-home task for an AI developer internship.

## Pipeline

1. `split_dataset.py` splits each person's photos into train and validation sets.
2. `baseline.py` extracts embeddings with InsightFace.
3. `baseline_evaluation.py` plots ROC, precision-recall and similarity distributions for the baseline.
4. `fine_tune.py` fine-tunes ResNet18 on the training set.
5. `evaluate_fine_tuned.py` produces the same plots for the fine-tuned model.

The results and plots are written up in `AI_Developer_Report_Vaikhashi.pdf`.

## Running it

```bash
python -m venv venv
venv\Scripts\activate        # on Windows
pip install -r requirements.txt
```

The dataset isn't in this repo. Put it in an `images/` folder, one folder per person with `High_quality/` and `Low_quality/` inside:

```
images/
  person1/
    High_quality/
    Low_quality/
```

Then run the scripts in the order above.
