"""Ứng dụng Streamlit so sánh dự đoán của baseline và DANN.

Nạp trọng số từ checkpoints/ và lịch sử huấn luyện đã lưu; không huấn luyện.

    streamlit run app.py
"""
import json
import random
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from da_demo.data import HALF_MEAN, get_dataset, preprocess
from da_demo.evaluate import summarize
from da_demo.models import DANN
from da_demo.train import lambda_at

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints"
MODELS = {
    "Mô hình cơ sở (chỉ học trên MNIST)": "baseline_mnist_to_mnist_m_gen.pt",
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


def seed_summary():
    # Trung bình và độ lệch chuẩn mẫu qua các seed trong runs/gen-half
    rows = {}
    for f in sorted((ROOT / "runs" / "gen-half").glob("seed*/*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        rows.setdefault(m["method"], []).append(m["target_test_acc"])
    return rows


@torch.no_grad()
def predict(model, mean, img: Image.Image):
    return torch.softmax(model(preprocess(img, mean).unsqueeze(0)), 1)[0]


st.set_page_config(page_title="MNIST → MNIST-M: mô hình cơ sở và DANN", layout="wide")
st.title("Thích ứng tên miền: MNIST → MNIST-M")
st.caption("So sánh mô hình chỉ học trên miền nguồn với DANN "
           "(Ganin & Lempitsky, ICML 2015; bản arXiv 2014).")

models = load_models()
if not models:
    st.error("Chưa có trọng số trong checkpoints/. Hãy chạy da_demo.train trước (xem README).")
    st.stop()

NAMES = {"baseline": "Mô hình cơ sở", "dann": "DANN", "target": "Cận trên (học trên MNIST-M)"}
rows = summarize(CKPT)
if rows:
    st.subheader("Độ chính xác trên tập kiểm thử")
    df = pd.DataFrame(rows)
    df["method"] = df["method"].map(NAMES).fillna(df["method"])
    st.dataframe(df[["method", "source_test_acc", "target_test_acc", "paper_target_acc", "epochs", "seed"]]
                 .rename(columns={"method": "Mô hình", "source_test_acc": "MNIST",
                                  "target_test_acc": "MNIST-M", "paper_target_acc": "MNIST-M (ICML)",
                                  "epochs": "Epoch", "seed": "Seed"}),
                 hide_index=True)
    runs = seed_summary()
    if runs:
        n = max(len(v) for v in runs.values())
        st.caption(f"Seed 42 là seed mặc định và có DANN cao nhất trong {n} seed. Trung bình ± độ lệch chuẩn mẫu "
                   f"trên MNIST-M qua {n} seed: " + "; ".join(
                       f"{NAMES.get(k, k)} {pd.Series(v).mean():.4f} ± {pd.Series(v).std():.4f}"
                       for k, v in runs.items()) + ".")

tab_pred, tab_hist = st.tabs(["Thử dự đoán", "Quá trình huấn luyện DANN"])

with tab_hist:
    dann = next((r for r in rows if r["method"] == "dann"), None)
    if dann and dann.get("history"):
        h = pd.DataFrame(dann["history"])
        total = dann.get("steps") or h["step"].max()
        h["lambda_p"] = [lambda_at(s / total) for s in h["step"]]
        h = h.set_index("epoch")
        st.markdown("Độ chính xác đo sau mỗi epoch (seed 42). Nhánh miền càng gần 0,5 thì hai miền càng khó phân biệt; "
                    "hệ số thích ứng λ_p tăng từ 0 đến 1 theo công thức (3.7).")
        st.line_chart(h[["source_test_acc", "target_test_acc", "domain_acc", "lambda_p"]].rename(columns={
            "source_test_acc": "MNIST (nguồn)", "target_test_acc": "MNIST-M (đích)",
            "domain_acc": "Nhánh miền", "lambda_p": "λ_p"}))
    else:
        st.info("Checkpoint DANN không có lịch sử huấn luyện.")

with tab_pred:
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
            col.metric("Dự đoán", int(probs.argmax()))
            col.caption(f"Xác suất lớp dự đoán: {probs.max().item():.1%}")
            col.bar_chart(pd.DataFrame({"xác suất": probs.numpy()}, index=[str(i) for i in range(10)]))
