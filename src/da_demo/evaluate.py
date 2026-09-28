"""Đo accuracy và tổng hợp kết quả các checkpoint thành bảng so sánh với bài báo.

    python -m da_demo.evaluate
"""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

# Bảng 1, Ganin & Lempitsky (2015), cột MNIST -> MNIST-M
PAPER = {"baseline": 0.5749, "dann": 0.8149, "target": 0.9891}


@torch.no_grad()
def accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / total


@torch.no_grad()
def domain_accuracy(model, src_loader, tgt_loader, device) -> float:
    """Accuracy của bộ phân loại miền: nguồn là 0, đích là 1; không dùng nhãn lớp."""
    model.eval()
    correct = total = 0
    for loader, dom in [(src_loader, 0), (tgt_loader, 1)]:
        for x, _ in loader:
            _, d = model(x.to(device), with_domain=True)
            correct += ((d > 0).long().cpu() == dom).sum().item()
            total += len(x)
    return correct / total


def summarize(ckpt_dir: Path = ROOT / "checkpoints") -> list[dict]:
    rows = []
    for f in sorted(ckpt_dir.glob("*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        m["paper_target_acc"] = PAPER.get(m["method"])
        rows.append(m)
    return rows


if __name__ == "__main__":
    print(f"{'method':<10}{'source acc':>12}{'target acc':>12}{'ICML 2015':>11}")
    for r in summarize():
        paper = f"{r['paper_target_acc']:.4f}" if r["paper_target_acc"] else "-"
        print(f"{r['method']:<10}{r['source_test_acc']:>12.4f}{r['target_test_acc']:>12.4f}{paper:>11}")
