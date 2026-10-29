import torch

@torch.no_grad()
def batch_metrics(logits, targets):
    predictions = torch.sigmoid(logits) >= 0.5
    targets = targets >= 0.5

    intersection = (predictions & targets).sum().item()
    pred_pixels = predictions.sum().item()
    target_pixels = targets.sum().item()

    union = pred_pixels + target_pixels - intersection

    return intersection, union, pred_pixels, target_pixels


def calculate_scores(totals):
    intersection, union, pred_pixels, target_pixels = totals

    iou = intersection / union if union else 1.0
    dice = (
        2 * intersection / (pred_pixels + target_pixels)
        if pred_pixels + target_pixels
        else 1.0
    )

    precision = (
        intersection / pred_pixels if pred_pixels else 0.0
    )
    recall = (
        intersection / target_pixels if target_pixels else 0.0
    )

    return iou, dice, precision, recall
