"""Ứng dụng Streamlit so sánh dự đoán của baseline và DANN.

Nạp trọng số từ checkpoints/.

    streamlit run app.py
"""
import random
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from da_demo.data import HALF_MEAN, get_dataset, preprocess
from da_demo.evaluate import summarize
from da_demo.models import DANN

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints"
MODELS = {
    "Baseline CNN (chỉ học trên MNIST)": "baseline_mnist_to_mnist_m_gen.pt",
    "DANN (MNIST + MNIST-M không nhãn)": "dann_mnist_to_mnist_m_gen.pt",
}


@st.cache_resource
def load_models():
    loaded = {}
    for label, fname in MODELS.items():
        path = CKPT / fname
        if path.exists():
            ckpt = torch.load(path, map_location="cpu")
            m = DANN()
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            # Checkpoint thiếu norm_mean dùng HALF_MEAN
            loaded[label] = (m, ckpt["metrics"].get("norm_mean", HALF_MEAN))
    return loaded


@st.cache_resource
def load_target_test():
    # MNIST-M dựng theo bài báo; tạo bằng python -m da_demo.make_mnistm
    try:
        return get_dataset("mnist_m_gen", ROOT / "data", train=False)
    except FileNotFoundError:
        return None


@torch.no_grad()
def predict(model, mean, img: Image.Image):
    return torch.softmax(model(preprocess(img, mean).unsqueeze(0)), 1)[0]


st.set_page_config(page_title="MNIST → MNIST-M: Baseline và DANN", layout="wide")
st.title("Thích ứng tên miền: MNIST → MNIST-M")
st.caption("So sánh mô hình chỉ học trên miền nguồn với DANN (Ganin & Lempitsky, 2015).")

models = load_models()
if not models:
    st.error("Chưa có trọng số trong checkpoints/. Hãy chạy da_demo.train trước (xem README).")
    st.stop()

rows = summarize(CKPT)
if rows:
    st.subheader("Accuracy trên tập kiểm thử")
    st.dataframe(pd.DataFrame(rows)[["method", "source_test_acc", "target_test_acc",
                                     "paper_target_acc", "epochs", "seed"]],
                 hide_index=True)

st.subheader("Thử dự đoán")
src = st.radio("Nguồn ảnh", ["Tải ảnh lên", "Ảnh ngẫu nhiên từ MNIST-M test"], horizontal=True)
img, true_label = None, None
if src == "Tải ảnh lên":
    up = st.file_uploader("Ảnh chữ số (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if up:
        img = Image.open(up)
else:
    ds = load_target_test()
    if ds is None:
        st.warning("Chưa có data/mnist_m_gen. Chạy: uv run python -m da_demo.make_mnistm")
    else:
        if st.button("Lấy ảnh khác") or "idx" not in st.session_state:
            st.session_state.idx = random.randrange(len(ds))
        img = Image.fromarray(ds.images[st.session_state.idx])
        true_label = ds.labels[st.session_state.idx]

if img is not None:
    cols = st.columns([1] + [2] * len(models))
    cols[0].image(img.convert("RGB").resize((112, 112), Image.NEAREST),
                  caption=f"Nhãn thật: {true_label}" if true_label is not None else None)
    for col, (label, (m, mean)) in zip(cols[1:], models.items()):
        probs = predict(m, mean, img)
        col.markdown(f"**{label}**")
        col.metric("Dự đoán", int(probs.argmax()), f"{probs.max().item():.1%}")
        col.bar_chart(pd.DataFrame({"xác suất": probs.numpy()}, index=[str(i) for i in range(10)]))
