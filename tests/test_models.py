import math

import torch

from da_demo.models import DANN, grad_reverse
from da_demo.train import lambda_at, lr_at, select_epoch


def test_output_shapes():
    m = DANN()
    x = torch.randn(4, 3, 28, 28)
    logits, d = m(x, lambd=1.0, with_domain=True)
    assert logits.shape == (4, 10)
    assert d.shape == (4,)


def test_grl_reverses_gradient():
    x = torch.randn(5, requires_grad=True)
    grad_reverse(x, 0.3).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.3))


def test_schedules_match_paper():
    assert lambda_at(0.0) == 0.0
    assert math.isclose(lambda_at(1.0), 2 / (1 + math.exp(-10)) - 1)
    assert math.isclose(lr_at(0.0), 0.01)
    assert math.isclose(lr_at(1.0), 0.01 / 11 ** 0.75)


def test_select_epoch_uses_no_target_labels():
    h = [{"epoch": 1, "source_test_acc": 0.99, "domain_acc": 0.90, "target_test_acc": 0.5},
         {"epoch": 2, "source_test_acc": 0.985, "domain_acc": 0.60, "target_test_acc": 0.1},
         {"epoch": 3, "source_test_acc": 0.95, "domain_acc": 0.50, "target_test_acc": 0.9}]
    # epoch 3 bị loại vì độ chính xác nguồn thấp hơn mức tốt nhất quá 0.01
    assert select_epoch(h)["epoch"] == 2
