# dann-mnist-mnistm: tái lập DANN trên MNIST → MNIST-M

Tái lập thí nghiệm MNIST → MNIST-M trong Ganin & Lempitsky (2015), *Unsupervised Domain Adaptation
by Backpropagation* ([arXiv:1409.7495](https://arxiv.org/abs/1409.7495)), và bản đầy đủ Ganin et al. (2016),
*Domain-Adversarial Training of Neural Networks* ([arXiv:1505.07818](https://arxiv.org/abs/1505.07818)).
Ba mô hình dùng chung một kiến trúc. Accuracy trên tập test MNIST-M (dựng lại theo bài báo), trung bình ± độ
lệch chuẩn mẫu trên 3 seed (42, 43, 44):

| Phương pháp | Dữ liệu huấn luyện | Ganin & Lempitsky (2015), Bảng 1 | Ganin et al. (2016), Bảng 2 | Đo lại, trừ 0.5 | Đo lại, trừ trung bình |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `baseline` | MNIST có nhãn | 0.5749 | 0.5225 | 0.5401 ± 0.0269 | 0.5144 ± 0.0335 |
| `dann`, epoch cuối | MNIST có nhãn + MNIST-M không nhãn | 0.8149 | 0.7666 | 0.8061 ± 0.0280 | 0.7996 ± 0.0091 |
| `dann`, epoch chọn không dùng nhãn đích | như trên | | | 0.7681 ± 0.0485 | 0.7533 ± 0.0571 |
| `target` | MNIST-M có nhãn (cận trên) | 0.9891 | 0.9596 | 0.9726 ± 0.0003 | 0.9730 ± 0.0006 |

DANN thu hẹp 61.5% (trừ 0.5) và 62.2% (trừ trung bình) khoảng cách giữa `baseline` và `target`, tính theo
`(dann - baseline) / (target - baseline)` trên độ chính xác trung bình; bản 2015 thu hẹp 57.9%. Không lần
huấn luyện nào sụp đổ. Kết quả từng lần chạy (có lịch sử theo epoch) ở `runs/gen-half` và `runs/gen-mean`;
GPU NVIDIA RTX A4000 Laptop, torch 2.14.0+cu130.

`checkpoints/` chứa ba mô hình seed 42, trừ 0.5: `baseline` 0.5309, `dann` 0.8266 (epoch cuối), `target`
0.9729. Ứng dụng Streamlit nạp các tệp này.

## Thiết lập theo bài báo

- Kiến trúc MNIST: conv 5×5 32 → max-pool → conv 5×5 48 → max-pool; nhánh nhãn FC 100 → 100 → 10;
  nhánh miền GRL → FC 100 → 1.
- SGD momentum 0.9, learning rate `μ_p = 0.01 / (1 + 10p)^0.75`.
- Hệ số GRL `λ_p = 2 / (1 + exp(-10p)) - 1`, `p` tăng tuyến tính từ 0 đến 1.
- Batch 128, một nửa nguồn, một nửa đích.
- MNIST-M: `I_out = |I1 - I2|`, với `I1` là chữ số MNIST và `I2` là mảnh cắt ngẫu nhiên từ ảnh màu BSDS500.

Chi tiết hai bản bài báo không nêu:

- Cách cắt mảnh BSDS500: mảnh 28×28 ở độ phân giải gốc; nền tập train lấy từ ảnh BSDS500 `train`, nền tập
  test lấy từ ảnh `test`; seed cắt mảnh 0.
- Tiền xử lý: bài báo ghi "mean subtraction". `--norm half` (mặc định) trừ 0.5 mỗi kênh; `--norm mean` trừ
  trung bình từng kênh trên tập train gộp của MNIST và MNIST-M.
- Số bước huấn luyện: đặt qua `--epochs` (mặc định 20 lượt duyệt tập nguồn); lịch lr và λ tính theo tổng số bước.
- Chọn mô hình: Ganin & Lempitsky (2015, mục 4) nhận xét thích ứng tốt hơn khi sai số test trên miền nguồn thấp
  và sai số bộ phân loại miền cao. Với DANN, `train.py` lưu thêm `*.selected.pt` là epoch có sai số bộ phân loại
  miền cao nhất trong các epoch có accuracy nguồn cách mức tốt nhất không quá 0.01; quy tắc không dùng nhãn đích.

## Dữ liệu

- MNIST: `torchvision.datasets.MNIST`, tự tải về `data/`.
- MNIST-M (mặc định `--target mnist_m_gen`): dựng bằng `python -m da_demo.make_mnistm` từ MNIST và
  [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) (tự tải,
  khoảng 70 MB), ghi `data/mnist_m_gen/{train,test}.npz` (60 000 ảnh train, 10 000 ảnh test).
- MNIST-M bản Hugging Face (`--target mnist_m`): [Mike0307/MNIST-M](https://huggingface.co/datasets/Mike0307/MNIST-M)
  (59 001 ảnh train, 9 001 ảnh test, 32×32 thu về 28×28, giấy phép MIT). Trên bản này DANN đạt 0.7146 / 0.7139
  / 0.7181 (trừ 0.5, seed 42 / 43 / 44; `runs/half`), thấp hơn rõ so với bản dựng lại.

## Cài đặt và chạy

```bash
git clone https://github.com/ThuanNP/dann-mnist-mnistm.git
cd dann-mnist-mnistm
uv sync                      # tạo .venv, cài torch, streamlit...
uv run pytest                # kiểm tra kiến trúc, GRL, lịch lr/λ, quy tắc chọn epoch
uv run python -m da_demo.make_mnistm                         # dựng MNIST-M

uv run python -m da_demo.train --method baseline --epochs 20
uv run python -m da_demo.train --method dann --epochs 20
uv run python -m da_demo.train --method target --epochs 20
uv run python -m da_demo.evaluate                            # bảng so sánh với bài báo

uv run streamlit run app.py
```

Mỗi lần huấn luyện lưu `checkpoints/<method>_mnist_to_mnist_m_gen.pt` (trọng số) và `.json` (accuracy, lịch sử
theo epoch, seed, số bước, thời gian huấn luyện, môi trường). Seed mặc định 42; đổi bằng `--seed`, đổi thư mục
ra bằng `--out-dir`.

## Notebook Colab

[`notebooks/dann_mnist_to_mnistm_colab.ipynb`](notebooks/dann_mnist_to_mnistm_colab.ipynb) là bản báo cáo tự
chứa: công thức của bài báo, dựng MNIST-M, mô tả dữ liệu, huấn luyện ba mô hình, bảng và hình kết quả (đường
cong theo epoch, t-SNE, ma trận nhầm lẫn). Mở trên Colab, chọn GPU T4, *Run all*; không cần cài thêm. Phần
huấn luyện nạp toàn bộ dữ liệu lên GPU thay cho `DataLoader`, nên số liệu gần với bảng trên nhưng không trùng
từng lần chạy.

## Môi trường tái lập

- Phiên bản thư viện khóa trong `uv.lock`. Trên Windows, `torch` và `torchvision` lấy bản CUDA 13.0 từ
  chỉ mục `https://download.pytorch.org/whl/cu130` (khai báo trong `pyproject.toml`); Linux và macOS
  cài từ PyPI.
- `seed.py` cố định seed cho `random`, NumPy, PyTorch, worker của DataLoader, và đặt
  `cudnn.deterministic = True`, `cudnn.benchmark = False`.
- Tệp `.json` ghi trường `environment`: phiên bản Python, hệ điều hành, `torch`, CUDA, cuDNN, tên GPU
  và commit git. Cùng seed nhưng khác GPU hoặc phiên bản CUDA, accuracy có thể lệch nhỏ.

## Cấu trúc

```
dann-mnist-mnistm/
├── app.py                 # Streamlit: nạp trọng số, tải ảnh lên, so sánh dự đoán
├── src/da_demo/
│   ├── data.py            # MNIST (3 kênh), MNIST-M, tiền xử lý dùng chung
│   ├── make_mnistm.py     # dựng MNIST-M theo bài báo
│   ├── models.py          # FeatureExtractor, LabelPredictor, DomainClassifier, GRL
│   ├── train.py           # huấn luyện baseline / dann / target, chọn epoch
│   ├── evaluate.py        # accuracy, bảng so sánh với bài báo
│   └── seed.py            # cố định seed
├── notebooks/             # notebook báo cáo trên Google Colab
├── runs/                  # kết quả từng seed (.json, log)
├── tests/test_models.py
└── checkpoints/           # trọng số đã huấn luyện
```

Trong `runs/`: `gen-half`, `gen-mean` là kết quả chính trên MNIST-M dựng lại; `half`, `mean` là cùng cấu hình
trên bản Hugging Face (lần chạy seed 43 với `--norm mean` sụp đổ, accuracy MNIST-M 0.1129).

## Giấy phép

Mã nguồn phát hành theo giấy phép MIT (xem `LICENSE`). MNIST-M bản Hugging Face theo giấy phép MIT của người
đăng. BSDS500 không đi kèm repo; `make_mnistm` tải từ trang của nhóm tác giả.
