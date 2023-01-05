# How to use

### Tên file helper và ý nghĩa
- `tf_cv_helper.py`: Các helper functions dùng cho các bàn toàn Computer Vision dùng Tensorflow
- `tf_nlp_helper.py`: Các helper functions dùng cho các bài toán xử lý ngôn ngữ tự nhiên dùng Tensorflow
- `pt_cv_helper.py`: Các helper functions dùng cho bàn toán Computer Vision dùng Pytorch
- `pt_nlp_helper.py`: Các helper functions dùng cho bài toán xử lý ngôn ngữ tự nhiên dùng Pytorch

### Download and import to your project

```python
if not os.path.exists("utils.py"):
    !wget https://raw.githubusercontent.com/ngohongthai/ml_helper/tensorflow-deep-learning/utils.py
else:
    print("[INFO] 'utils.py' already exists, skipping download.")

# If you want to modify ..._helper.py
%reload_ext autoreload
%autoreload 2
```