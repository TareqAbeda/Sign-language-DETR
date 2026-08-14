"""
Per-class evaluation for the sign-gesture DETR model.

Drop this file into the same folder as test.py (i.e. src/) and run:
    uv run evaluate.py

For each image in the test set, it uses the same HungarianMatcher that
train.py/loss.py already use to optimally pair each ground-truth box with
one predicted query. It then checks whether that matched query's predicted
class equals the true class, and aggregates the results per gesture class.

This gives you the numbers the "Training Results" section of the report
needs: per-class accuracy, overall accuracy, a confusion matrix, and mean
detection confidence per class.
"""

import torch
from collections import defaultdict

from data import DETRData
from model import DETR
from loss import HungarianMatcher
from torch.utils.data import DataLoader
from utils.boxes import stacker
from utils.setup import get_classes

CLASSES = get_classes()
NUM_CLASSES = len(CLASSES)

# Same weighting used for training/matching in train.py -- keep in sync if you change it there.
WEIGHT_DICT = {'class_weighting': 1, 'bbox_weighting': 5, 'giou_weighting': 2}


def evaluate(checkpoint='pretrained/99_model.pt', confidence_threshold=0.5):
    test_dataset = DETRData('data/test', train=False)
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=4, collate_fn=stacker, drop_last=False
    )

    model = DETR(num_classes=NUM_CLASSES)
    model.eval()
    model.load_pretrained(checkpoint)

    matcher = HungarianMatcher(WEIGHT_DICT)

    # Per-class tallies
    total_per_class = defaultdict(int)
    correct_per_class = defaultdict(int)
    confidence_sum_per_class = defaultdict(float)
    below_threshold_per_class = defaultdict(int)  # matched, right class, but low confidence
    confusion = defaultdict(lambda: defaultdict(int))  # confusion[true_class][predicted_class] += 1

    with torch.no_grad():
        for X, y in test_dataloader:
            result = model(X)
            probabilities = result['pred_logits'].softmax(-1)  # [B, num_queries, num_classes+1]

            # Optimally match each ground-truth box to one predicted query, per image
            indices = matcher(result, y)

            for batch_idx, (query_idx, target_idx) in enumerate(indices):
                true_labels = y[batch_idx]['labels']
                for q, t in zip(query_idx.tolist(), target_idx.tolist()):
                    true_class = true_labels[t].item()
                    pred_probs = probabilities[batch_idx, q]
                    pred_class = pred_probs[:-1].argmax().item()  # exclude "no object" class
                    pred_conf = pred_probs[pred_class].item()

                    true_name = CLASSES[true_class]
                    total_per_class[true_name] += 1
                    confusion[true_name][CLASSES[pred_class]] += 1

                    if pred_class == true_class:
                        correct_per_class[true_name] += 1
                        confidence_sum_per_class[true_name] += pred_conf
                        if pred_conf < confidence_threshold:
                            below_threshold_per_class[true_name] += 1

    # ---- Report ----
    print(f"\n{'Class':<15}{'Total':>8}{'Correct':>10}{'Accuracy':>12}{'Avg Conf':>12}")
    print("-" * 57)
    total_all, correct_all = 0, 0
    for cls in CLASSES:
        total = total_per_class[cls]
        correct = correct_per_class[cls]
        acc = (correct / total * 100) if total else 0.0
        avg_conf = (confidence_sum_per_class[cls] / correct) if correct else 0.0
        print(f"{cls:<15}{total:>8}{correct:>10}{acc:>11.1f}%{avg_conf:>12.2f}")
        total_all += total
        correct_all += correct

    overall_acc = (correct_all / total_all * 100) if total_all else 0.0
    print("-" * 57)
    print(f"{'OVERALL':<15}{total_all:>8}{correct_all:>10}{overall_acc:>11.1f}%")

    print("\nConfusion matrix (rows = true class, columns = predicted class):")
    header = " " * 15 + "".join(f"{c:>12}" for c in CLASSES)
    print(header)
    for true_cls in CLASSES:
        row = "".join(f"{confusion[true_cls][pred_cls]:>12}" for pred_cls in CLASSES)
        print(f"{true_cls:<15}{row}")

    return {
        'total_per_class': dict(total_per_class),
        'correct_per_class': dict(correct_per_class),
        'overall_accuracy': overall_acc,
    }


if __name__ == '__main__':
    evaluate()


    