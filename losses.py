

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
   

    def __init__(self, num_classes=2, weight=None, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        if weight is None:
            weight = torch.ones(num_classes)
        else:
            weight = torch.as_tensor(weight, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_onehot, dim=dims)
        cardinality = torch.sum(probs + target_onehot, dim=dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss_per_class = 1.0 - dice_per_class

        w = self.weight.to(logits.device)
        return (loss_per_class * w).sum() / w.sum()


class BCEDiceLoss(nn.Module):
   

    def __init__(self, class_weights=(0.2, 0.8), num_classes=2):
        super().__init__()
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        self.bce = nn.CrossEntropyLoss(weight=weight_tensor)
        self.dice = DiceLoss(num_classes=num_classes, weight=weight_tensor)

    def forward(self, logits, target):
        return self.bce(logits, target) + self.dice(logits, target)
