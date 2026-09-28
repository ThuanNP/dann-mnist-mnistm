"""Nạp dữ liệu MNIST (miền nguồn) và MNIST-M (miền đích).

- MNIST: tải qua torchvision.
- MNIST-M: bộ dữ liệu do Ganin & Lempitsky (2015) tạo bằng cách trộn chữ số MNIST với
  mảnh ảnh màu từ BSDS500. Có hai bản:
  - `mnist_m_gen` (mặc định): dựng lại theo công thức của bài báo bằng `da_demo.make_mnistm`
    (60.000 ảnh train, 10.000 ảnh test).
  - `mnist_m`: bản trên Hugging Face https://huggingface.co/datasets/Mike0307/MNIST-M
    (59.001 ảnh train, 9.001 ảnh test, giấy phép MIT).
"""
import io
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

MNISTM_REPO = "Mike0307/MNIST-M"
MNISTM_FILES = {
    "train": "data/train-00000-of-00001-571b6b1e2c195186.parquet",
    "test": "data/test-00000-of-00001-ba3ad971b105ff65.parquet",
}

IMG_SIZE = 28

# Mặc định trừ 0.5 mỗi kênh, giữ nguyên thang đo.
HALF_MEAN = (0.5, 0.5, 0.5)


@lru_cache(maxsize=None)
def _transform(mean: tuple) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=(1.0, 1.0, 1.0)),
    ])


def to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


def preprocess(img: Image.Image, mean=HALF_MEAN) -> torch.Tensor:
    """Tiền xử lý một ảnh bất kỳ (dùng chung cho huấn luyện và ứng dụng Streamlit)."""
    return _transform(tuple(mean))(to_rgb(img))


def train_channel_mean(root: Path, target: str = "mnist_m") -> tuple:
    """Trung bình từng kênh (thang [0, 1]) trên tập train gộp của MNIST và bộ MNIST-M `target`."""
    mnist = datasets.MNIST(root=str(root), train=True, download=True).data.float().div(255)
    sums = torch.full((3,), mnist.sum().item())
    count = mnist.numel()
    tgt = MNISTMGen(root, train=True) if target == "mnist_m_gen" else MNISTM(root, train=True)
    for img, _ in tgt.raw():
        x = transforms.functional.to_tensor(to_rgb(img).resize((IMG_SIZE, IMG_SIZE),
                                                               Image.BILINEAR))
        sums += x.sum(dim=(1, 2))
        count += IMG_SIZE * IMG_SIZE
    return tuple(round(v, 6) for v in (sums / count).tolist())


class MNISTRGB(Dataset):
    """MNIST chuyển sang 3 kênh để dùng chung kiến trúc với MNIST-M."""

    def __init__(self, root: Path, train: bool, mean=HALF_MEAN):
        self.ds = datasets.MNIST(root=str(root), train=train, download=True)
        self.mean = tuple(mean)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        return preprocess(img, self.mean), label


class MNISTM(Dataset):
    def __init__(self, root: Path, train: bool, mean=HALF_MEAN):
        self.mean = tuple(mean)
        from huggingface_hub import hf_hub_download

        split = "train" if train else "test"
        path = hf_hub_download(
            repo_id=MNISTM_REPO,
            filename=MNISTM_FILES[split],
            repo_type="dataset",
            local_dir=str(Path(root) / "mnist_m"),
        )
        df = pd.read_parquet(path)
        self.images = [row["bytes"] for row in df["image"]]
        self.labels = df["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def raw(self):
        for b, y in zip(self.images, self.labels):
            yield Image.open(io.BytesIO(b)), y

    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self.images[idx]))
        return preprocess(img, self.mean), self.labels[idx]


class MNISTMGen(Dataset):
    """MNIST-M tạo bằng `da_demo.make_mnistm` (ảnh 28x28, 60.000 train, 10.000 test)."""

    def __init__(self, root: Path, train: bool, mean=HALF_MEAN):
        path = Path(root) / "mnist_m_gen" / ("train.npz" if train else "test.npz")
        if not path.exists():
            raise FileNotFoundError(f"Thiếu {path}; chạy python -m da_demo.make_mnistm")
        d = np.load(path)
        self.images, self.labels, self.mean = d["images"], d["labels"].tolist(), tuple(mean)

    def __len__(self) -> int:
        return len(self.labels)

    def raw(self):
        for x, y in zip(self.images, self.labels):
            yield Image.fromarray(x), y

    def __getitem__(self, idx):
        return preprocess(Image.fromarray(self.images[idx]), self.mean), self.labels[idx]


def get_dataset(name: str, root: Path, train: bool, mean=HALF_MEAN) -> Dataset:
    if name == "mnist":
        return MNISTRGB(root, train, mean)
    if name == "mnist_m":
        return MNISTM(root, train, mean)
    if name == "mnist_m_gen":
        return MNISTMGen(root, train, mean)
    raise ValueError(f"Không có bộ dữ liệu {name}")
