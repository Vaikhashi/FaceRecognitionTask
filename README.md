# Face recognition on low-quality images

Face matching usually works fine on clear photos and falls apart on blurry, dark or tiny ones. This project looks at how much a model can recover on the low-quality ones.

I started with a pretrained InsightFace model to get face embeddings as a baseline, then fine-tuned a ResNet18 on the same dataset and compared the two. I built it as a take-home task for an AI developer internship.

## Pipeline

1. `split_dataset.py` splits each person's photos into train and validation sets.
2. `baseline.py` extracts embeddings with InsightFace.
3. `baseline_evaluation.py` plots ROC, precision-recall and similarity distributions for the baseline.
4. `fine_tune.py` fine-tunes ResNet18 on the training set.
5. `evaluate_fine_tuned.py` produces the same plots for the fine-tuned model.

## Results

**Baseline.** I tested the pretrained InsightFace model on 10 photos of 5 people (one high-quality and one low-quality photo each), using cosine similarity with a match threshold of 0.5.

- 4 of the 5 same-person pairs (high vs low quality) matched, with scores between 0.50 and 0.60. The one miss scored 0.36.
- 2 of the 40 different-person pairs were wrongly matched (0.51 and 0.53), and both were two low-quality photos.

So the pretrained model mostly works, but low-quality photos of different people start to look alike. That's the gap fine-tuning was meant to close.

**Fine-tuned.** ResNet18 (pretrained on ImageNet) with a 128-d embedding head, trained with Adam and cross-entropy on about 9,000 training images. Corrupted images are skipped instead of crashing the run. On the validation set, the ROC and precision-recall curves improved, and same-person and different-person scores separated more clearly.

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
