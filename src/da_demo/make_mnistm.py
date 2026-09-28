"""Tạo MNIST-M theo mô tả của Ganin et al. (2016, JMLR), mục 5.2.4.

Mỗi ảnh là I_out[i, j, k] = |I1[i, j, k] - I2[i, j, k]|, với I1 là chữ số MNIST (lặp thành 3 kênh) và
I2 là mảnh cắt ngẫu nhiên từ ảnh màu BSDS500 (Arbelaez et al., 2011). Bài báo không nêu thêm chi tiết; ở đây
mảnh 28x28 cắt ở độ phân giải gốc, nền của tập train lấy từ ảnh BSDS500 `train`, nền của tập test lấy từ
ảnh BSDS500 `test`.

    python -m da_demo.make_mnistm            # ghi data/mnist_m_gen/{train,test}.npz
"""
import argparse
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import datasets

BSDS_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
ROOT = Path(__file__).resolve().parents[2]
SIZE = 28


def bsds_images(data_dir: Path, split: str) -> list:
    tgz = data_dir / "BSR_bsds500.tgz"
    if not tgz.exists():
        urllib.request.urlretrieve(BSDS_URL, tgz)
    out = []
    with tarfile.open(tgz) as tar:
        for m in tar.getmembers():
            if f"BSDS500/data/images/{split}/" in m.name and m.name.endswith(".jpg"):
                out.append(np.asarray(Image.open(tar.extractfile(m)).convert("RGB")))
    return out


def blend(digits: np.ndarray, backgrounds: list, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((len(digits), SIZE, SIZE, 3), dtype=np.uint8)
    for n, d in enumerate(digits):
        bg = backgrounds[rng.integers(len(backgrounds))]
        y = rng.integers(bg.shape[0] - SIZE + 1)
        x = rng.integers(bg.shape[1] - SIZE + 1)
        patch = bg[y:y + SIZE, x:x + SIZE].astype(np.int16)
        out[n] = np.abs(d[:, :, None].astype(np.int16) - patch).astype(np.uint8)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    out_dir = args.data_dir / "mnist_m_gen"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, bsds_split in [("train", "train"), ("test", "test")]:
        mnist = datasets.MNIST(root=str(args.data_dir), train=split == "train", download=True)
        images = blend(mnist.data.numpy(), bsds_images(args.data_dir, bsds_split), rng)
        np.savez_compressed(out_dir / f"{split}.npz", images=images, labels=mnist.targets.numpy())
        print(split, images.shape)


if __name__ == "__main__":
    main()
