"""Ứng dụng Streamlit so sánh dự đoán của baseline và DANN.

Nạp trọng số từ checkpoints/ và lịch sử huấn luyện đã lưu; không huấn luyện.

    streamlit run app.py
"""
import json
import random
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

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
    # Độ chính xác MNIST-M theo seed của từng phương pháp trong runs/gen-half
    rows = {}
    for f in sorted((ROOT / "runs" / "gen-half").glob("seed*/*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        rows.setdefault(m["method"], {})[m["seed"]] = m["target_test_acc"]
    return rows


@torch.no_grad()
def predict(model, mean, img: Image.Image):
    return torch.softmax(model(preprocess(img, mean).unsqueeze(0)), 1)[0]


st.set_page_config(page_title="MNIST → MNIST-M: mô hình cơ sở và DANN", layout="wide")
# Cỡ chữ gốc 17px (mặc định 16px) cho dễ đọc khi trình chiếu; giữ theme theo hệ thống.
# Nút chọn nguồn ảnh cao bằng nút thường (2.5rem) để thẳng hàng với "Lấy ảnh khác".
st.html("<style>html { font-size: 17px; } .st-key-src button { min-height: 2.5rem; }</style>")
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
    acc = ["source_test_acc", "target_test_acc", "paper_target_acc"]
    df[acc] = df[acc] * 100
    pct = st.column_config.NumberColumn(format="%.2f%%")
    st.dataframe(df[["method", "source_test_acc", "target_test_acc", "paper_target_acc", "epochs", "seed"]]
                 .rename(columns={"method": "Mô hình", "source_test_acc": "MNIST",
                                  "target_test_acc": "MNIST-M", "paper_target_acc": "MNIST-M (ICML)",
                                  "epochs": "Epoch", "seed": "Seed"}),
                 column_config={"MNIST": pct, "MNIST-M": pct, "MNIST-M (ICML)": pct},
                 hide_index=True)
    runs = seed_summary()
    if runs:
        n = max(len(v) for v in runs.values())
        seed = rows[0]["seed"]
        note = f"Seed {seed} là seed mặc định"
        if "dann" in runs and max(runs["dann"], key=runs["dann"].get) == seed:
            note += f" và có DANN cao nhất trong {n} seed"
        st.caption(note + f". Trung bình ± độ lệch chuẩn mẫu trên MNIST-M qua {n} seed: " + "; ".join(
            f"{NAMES.get(k, k)} {pd.Series(v).mean():.2%} ± {pd.Series(v).std():.2%}"
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
    # Nút chọn nguồn và nút "Lấy ảnh khác" chung một hàng; tự xuống dòng khi màn hình hẹp
    row = st.container(horizontal=True, vertical_alignment="bottom")
    src = row.segmented_control("Nguồn ảnh", ["Tải ảnh lên", "Ảnh ngẫu nhiên từ MNIST-M test"],
                                default="Tải ảnh lên", required=True, key="src")
    img, true_label = None, None
    if src == "Tải ảnh lên":
        up = st.file_uploader("Ảnh chữ số (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if up:
            try:
                img = Image.open(up)
                img.load()
            except (UnidentifiedImageError, OSError):
                img = None
                st.error("Không đọc được tệp ảnh. Hãy chọn một ảnh PNG hoặc JPG khác.")
    else:
        ds = load_target_test()
        if ds is None:
            st.warning("Chưa có data/mnist_m_gen. Chạy: uv run python -m da_demo.make_mnistm")
        else:
            if row.button("Lấy ảnh khác", icon=":material/shuffle:") or "idx" not in st.session_state:
                st.session_state.idx = random.randrange(len(ds))
            img = Image.fromarray(ds.images[st.session_state.idx])
            true_label = ds.labels[st.session_state.idx]

    if img is not None:
        cols = st.columns([1.2] + [2] * len(models), gap="large")
        with cols[0].container(border=True):
            st.markdown("**Ảnh đầu vào**")
            st.image(img.convert("RGB").resize((224, 224), Image.NEAREST), width=224)
            if true_label is not None:
                st.markdown(f"Nhãn thật: **{true_label}**")
        for col, (label, (m, mean)) in zip(cols[1:], models.items()):
            probs = predict(m, mean, img)
            pred = int(probs.argmax())
            with col.container(border=True):
                verdict = "" if true_label is None else (
                    "<span style='color:#21c354;font-size:1.3rem;font-weight:600'>đúng</span>" if pred == true_label
                    else "<span style='color:#ff4b4b;font-size:1.3rem;font-weight:600'>sai</span>")
                # Lớp khác: xanh dương nhạt; lớp dự đoán khi chưa biết nhãn: xanh dương đậm; đúng: xanh lá; sai: đỏ
                accent = "#1c6fd1" if true_label is None else ("#21c354" if pred == true_label else "#ff4b4b")
                # Tên mô hình và kết quả dự đoán chung một hàng để biểu đồ có thêm chỗ
                head = st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center")
                head.markdown(f"**{label}**", width="content")
                head.markdown(
                    "<div style='display:flex;align-items:baseline;gap:1.1rem'>"
                    "<span style='font-size:1.3rem;opacity:.75'>Dự đoán</span>"
                    f"<span style='font-size:2.8rem;font-weight:700;line-height:1'>{pred}</span>"
                    f"<span style='font-size:1.3rem'>{probs.max().item():.1%}</span>{verdict}</div>",
                    unsafe_allow_html=True, width="content")
                # Cột lớp dự đoán tô màu nhấn; nhãn trục x đặt đứng
                df_p = pd.DataFrame({"chữ số": [str(i) for i in range(10)], "xác suất": probs.numpy(),
                                     "dự đoán": [i == pred for i in range(10)]})
                st.altair_chart(alt.Chart(df_p, height=320).mark_bar(cornerRadiusTopLeft=3,
                                                                     cornerRadiusTopRight=3).encode(
                    x=alt.X("chữ số:N", title="Chữ số",
                            axis=alt.Axis(labelAngle=0, labelFontSize=15, titleFontSize=14)),
                    y=alt.Y("xác suất:Q", title="Xác suất", scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(format="%", labelFontSize=13, titleFontSize=14)),
                    color=alt.condition("datum['dự đoán']", alt.value(accent), alt.value("#a8cdf0")),
                    tooltip=["chữ số", alt.Tooltip("xác suất:Q", format=".1%")]))
