"""Kiến trúc cho MNIST ("MNIST architecture") trong Ganin & Lempitsky (2015).

Bộ trích đặc trưng: conv 5x5 32 -> ReLU -> max-pool 2x2 -> conv 5x5 48 -> ReLU -> max-pool 2x2.
Bộ dự đoán nhãn: FC 100 -> ReLU -> FC 100 -> ReLU -> FC 10.
Bộ phân loại miền: GRL -> FC 100 -> ReLU -> FC 1 (logistic).
"""
import torch
from torch import nn


class GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer: lan truyền xuôi giữ nguyên, lan truyền ngược nhân -lambda."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 48, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.out_dim = 48 * 4 * 4

    def forward(self, x):
        return self.net(x)


class LabelPredictor(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 100), nn.ReLU(),
            nn.Linear(100, 100), nn.ReLU(),
            nn.Linear(100, n_classes),
        )

    def forward(self, f):
        return self.net(f)


class DomainClassifier(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 100), nn.ReLU(), nn.Linear(100, 1))

    def forward(self, f, lambd: float):
        return self.net(grad_reverse(f, lambd)).squeeze(1)


class DANN(nn.Module):
    """Dùng chung cho baseline (bỏ qua nhánh miền) và DANN."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = FeatureExtractor()
        self.classifier = LabelPredictor(self.features.out_dim, n_classes)
        self.domain = DomainClassifier(self.features.out_dim)

    def forward(self, x, lambd: float = 0.0, with_domain: bool = False):
        f = self.features(x)
        logits = self.classifier(f)
        if with_domain:
            return logits, self.domain(f, lambd)
        return logits
