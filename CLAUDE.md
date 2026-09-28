# CLAUDE.md

Tái lập DANN (Ganin & Lempitsky, 2015) trên MNIST → MNIST-M. Repo công khai: không đưa ghi chú nội bộ,
mã Zotero hay thông tin cá nhân vào đây; chạy `gitleaks git --pre-commit --staged .` trước khi commit.

## Lệnh

- `uv sync`; `uv run pytest -q` (4 test: kích thước đầu ra, GRL, lịch lr/λ, quy tắc chọn epoch)
- `uv run python -m da_demo.make_mnistm`: dựng MNIST-M theo bài báo vào `data/mnist_m_gen` (mặc định của `--target`)
- `uv run python -m da_demo.train --method baseline|dann|target --epochs N`; chạy thử nhanh bằng `--max-steps 50`
- `uv run python -m da_demo.evaluate`; `uv run streamlit run app.py`

## Quy ước

- Siêu tham số theo mục 4 và Phụ lục C của Ganin & Lempitsky (2015); chi tiết bài báo không nêu ghi ở README.
- `data.preprocess` dùng chung cho huấn luyện và app; sửa tiền xử lý ở đó để hai phía không lệch nhau.
- App chỉ nạp `checkpoints/*.pt`, không huấn luyện. Checkpoint và `.json` kết quả được commit.
- Notebook `notebooks/dann_mnist_to_mnistm_colab.ipynb` tự chứa (không import `da_demo`); sửa thiết lập huấn luyện
  thì sửa cả notebook.
- Chỉ ghi số liệu đo được từ lần chạy thật; mốc trên MNIST-M (baseline / dann / target): bản ICML 2015
  0.5749 / 0.8149 / 0.9891, bản JMLR 2016 0.5225 / 0.7666 / 0.9596.
- README, docstring, comment, notebook viết tiếng Việt; không tự tham chiếu, tự phủ định, tự biện minh.
