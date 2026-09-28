"""Huấn luyện baseline (chỉ miền nguồn), DANN, hoặc mô hình cận trên (huấn luyện trên đích).

Thiết lập theo Ganin & Lempitsky (2015), mục 4 và Phụ lục C:
- SGD, momentum 0.9; learning rate mu_p = mu_0 / (1 + alpha * p)^beta, mu_0 = 0.01, alpha = 10, beta = 0.75.
- Hệ số thích ứng lambda_p = 2 / (1 + exp(-gamma * p)) - 1, gamma = 10.
- Batch 128: một nửa từ miền nguồn (có nhãn), một nửa từ miền đích (không dùng nhãn).

Ví dụ:
    python -m da_demo.train --method baseline --epochs 20
    python -m da_demo.train --method dann --epochs 20
"""
import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import HALF_MEAN, get_dataset, train_channel_mean
from .evaluate import accuracy, domain_accuracy
from .models import DANN
from .seed import seed_worker, set_seed

ROOT = Path(__file__).resolve().parents[2]


def environment(device):
    """Môi trường chạy: Python, hệ điều hành, torch, CUDA, cuDNN, GPU, commit git."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "git_commit": commit,
    }


def lr_at(p, mu0=0.01, alpha=10.0, beta=0.75):
    return mu0 / (1.0 + alpha * p) ** beta


def lambda_at(p, gamma=10.0):
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


def select_epoch(history, tol=0.01):
    """Chọn epoch không dùng nhãn đích, theo nhận xét ở mục 4 của Ganin & Lempitsky (2015):
    thích ứng tốt hơn khi sai số kiểm thử trên miền nguồn thấp và sai số bộ phân loại miền cao.
    Trong các epoch có độ chính xác nguồn cách mức tốt nhất không quá `tol`, lấy epoch có sai số
    bộ phân loại miền (1 - domain_acc) cao nhất; trùng thì lấy epoch sau."""
    best = max(h["source_test_acc"] for h in history)
    ok = [h for h in history if h["source_test_acc"] >= best - tol]
    h = max(ok, key=lambda h: (1 - h["domain_acc"], h["epoch"]))
    return {"epoch": h["epoch"], "rule": f"source_acc >= best - {tol}; max domain error",
            "source_test_acc": h["source_test_acc"], "domain_acc": h["domain_acc"],
            "target_test_acc": h["target_test_acc"]}


def make_loader(ds, batch, gen, shuffle=True, workers=2):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=shuffle,
                      num_workers=workers, worker_init_fn=seed_worker, generator=gen,
                      pin_memory=torch.cuda.is_available())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["baseline", "dann", "target"], required=True)
    ap.add_argument("--source", default="mnist")
    ap.add_argument("--target", default="mnist_m_gen")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128, help="tổng kích thước batch nguồn + đích")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=None, help="giới hạn số bước (chạy thử)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--norm", choices=["half", "mean"], default="half",
                    help="half: trừ 0.5 mỗi kênh; mean: trừ trung bình kênh của tập train hai miền")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "checkpoints")
    args = ap.parse_args(argv)

    gen = set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half = args.batch // 2

    mean = HALF_MEAN if args.norm == "half" else train_channel_mean(args.data_dir, args.target)
    src_train = get_dataset(args.source, args.data_dir, train=True, mean=mean)
    tgt_train = get_dataset(args.target, args.data_dir, train=True, mean=mean)
    src_test = get_dataset(args.source, args.data_dir, train=False, mean=mean)
    tgt_test = get_dataset(args.target, args.data_dir, train=False, mean=mean)

    # Với "target", mô hình học có giám sát trên miền đích (cận trên trong Bảng 1 của Ganin & Lempitsky, 2015)
    labeled = tgt_train if args.method == "target" else src_train
    bs_labeled = args.batch if args.method != "dann" else half
    lab_loader = make_loader(labeled, bs_labeled, gen, workers=args.workers)
    tgt_loader = make_loader(tgt_train, half, gen, workers=args.workers)

    model = DANN().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Đánh giá theo epoch dùng Generator riêng để không đổi thứ tự dữ liệu huấn luyện
    eval_gen = torch.Generator().manual_seed(args.seed)
    ev = lambda ds: accuracy(model, make_loader(ds, 512, eval_gen, shuffle=False, workers=0), device)
    history, eval_seconds, snapshots = [], [0.0], {}

    def log_epoch(epoch):
        t = time.time()
        rec = {"epoch": epoch, "step": step, "source_test_acc": ev(src_test)}
        if args.method == "dann":
            rec["domain_acc"] = domain_accuracy(
                model, make_loader(src_test, 512, eval_gen, shuffle=False, workers=0),
                make_loader(tgt_test, 512, eval_gen, shuffle=False, workers=0), device)
        rec["target_test_acc"] = ev(tgt_test)  # chỉ để phân tích, không dùng chọn cấu hình
        history.append(rec)
        snapshots[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.train()
        eval_seconds[0] += time.time() - t

    steps_per_epoch = len(lab_loader)
    total = steps_per_epoch * args.epochs
    if args.max_steps:
        total = min(total, args.max_steps)
    step = 0
    t0 = time.time()
    model.train()
    while step < total:
        tgt_iter = iter(tgt_loader)
        for xs, ys in lab_loader:
            if step >= total:
                break
            p = step / total
            for g in opt.param_groups:
                g["lr"] = lr_at(p)
            xs, ys = xs.to(device), ys.to(device)

            if args.method == "dann":
                try:
                    xt, _ = next(tgt_iter)  # bỏ nhãn đích
                except StopIteration:
                    tgt_iter = iter(tgt_loader)
                    xt, _ = next(tgt_iter)
                xt = xt.to(device)
                lambd = lambda_at(p)
                x = torch.cat([xs, xt])
                d = torch.cat([torch.zeros(len(xs)), torch.ones(len(xt))]).to(device)
                logits, d_logits = model(x, lambd=lambd, with_domain=True)
                loss_y = F.cross_entropy(logits[: len(xs)], ys)
                loss_d = F.binary_cross_entropy_with_logits(d_logits, d)
                loss = loss_y + loss_d
            else:
                loss = loss_y = F.cross_entropy(model(xs), ys)
                loss_d = torch.tensor(0.0)

            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 200 == 0 or step == total:
                print(f"[{args.method}] step {step}/{total} loss_y={loss_y.item():.4f} "
                      f"loss_d={loss_d.item():.4f} lr={lr_at(p):.5f} ({time.time() - t0:.0f}s)")
        if step % steps_per_epoch == 0 or step == total:
            log_epoch(-(-step // steps_per_epoch))

    train_seconds = time.time() - t0 - eval_seconds[0]
    selected = select_epoch(history) if args.method == "dann" else None
    model.eval()
    metrics = {
        "method": args.method, "source": args.source, "target": args.target,
        "seed": args.seed, "epochs": args.epochs, "steps": total,
        "norm": args.norm, "norm_mean": list(mean),
        "source_test_acc": ev(src_test), "target_test_acc": ev(tgt_test),
        "train_seconds": round(train_seconds, 1), "environment": environment(device),
        "history": history, "selected": selected,
    }
    print(json.dumps(metrics, indent=2))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.method}_{args.source}_to_{args.target}"
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, args.out_dir / f"{name}.pt")
    (args.out_dir / f"{name}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if selected:
        torch.save({"state_dict": snapshots[selected["epoch"]], "metrics": {**metrics, **selected}},
                   args.out_dir / f"{name}.selected.pt")


if __name__ == "__main__":
    main()
