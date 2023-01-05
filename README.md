# How to use

### Tên file helper và ý nghĩa
- [tf_cv_helper.py](https://raw.githubusercontent.com/ngohongthai/ml_helper/main/tf_cv_helper.py): Các helper functions dùng cho các bàn toàn Computer Vision dùng Tensorflow
- [tf_nlp_helper.py](https://raw.githubusercontent.com/ngohongthai/ml_helper/main/tf_nlp_helper.py): Các helper functions dùng cho các bài toán xử lý ngôn ngữ tự nhiên dùng Tensorflow
- [pt_cv_helper.py](https://raw.githubusercontent.com/ngohongthai/ml_helper/main/pt_cv_helper.py): Các helper functions dùng cho bàn toán Computer Vision dùng Pytorch
- [pt_nlp_helper.py](https://raw.githubusercontent.com/ngohongthai/ml_helper/main/pt_nlp_helper.py): Các helper functions dùng cho bài toán xử lý ngôn ngữ tự nhiên dùng Pytorch
- Các file `...notebook.ipynb` là các ví dụ tương ứng

### Download and import vào trong project

Ví dụ: 
```python
if not os.path.exists("tf_cv_helper.py"):
    !wget https://raw.githubusercontent.com/ngohongthai/ml_helper/main/tf_cv_helper.py
else:
    print("[INFO] 'tf_cv_helper.py' already exists, skipping download.")

import tf_cv_helper as helper

# Sử dụng trong file notebook để auto reload những thay đổi có trong file helper (nếu có)
%reload_ext autoreload
%autoreload 2
```