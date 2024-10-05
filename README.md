# Hand Gesture Detection

## Mô tả dự án

Dự án Hand Gesture Detection là một hệ thống nhận diện cử chỉ tay dựa trên AI, sử dụng PyTorch và dữ liệu từ camera để phát hiện và nhận diện các cử chỉ tay trong thời gian thực. Hệ thống sử dụng mô hình học sâu để học các đặc trưng từ hình ảnh tay (21 điểm đặc trưng) và phân loại chúng thành các nhãn cử chỉ khác nhau.

### Mục tiêu

- Phát hiện các cử chỉ tay phổ biến (mở tay, nắm tay, ngón cái, ký hiệu hòa bình, v.v.).
- Sử dụng dữ liệu từ camera để nhận diện cử chỉ tay thời gian thực.
- Cung cấp mã nguồn đầy đủ và cấu trúc rõ ràng để mở rộng và tích hợp vào các hệ thống khác.

---

## Cấu trúc thư mục dự án

motion_detect/
├── src/
│ ├── core/
│ │ ├── capture.py
│ │ ├── train.py
│ │ ├── detect.py
│ ├── models/
│ │ ├── hand_gesture_model.py
│ ├── utils/
│ │ ├── file_utils.py
│ │ ├── data_utils.py
│ │ ├── hand_utils.py
│ │ ├── video_utils.py
│ ├── services/
│ │ ├── gesture_recognition_service.py
│ │ ├── camera_service.py
│ └── configs/
│ ├── labels.csv
│ └── settings.py
├── requirements.txt
└── README.md

---

## Cài đặt

### 1. Yêu cầu hệ thống:

- Python >= 3.7
- PyTorch >= 1.7.0
- OpenCV >= 4.5.0
- Pandas >= 1.1.0
- NumPy >= 1.18.0

### 2. Tạo môi trường ảo và cài đặt thư viện

```bash
# Tạo môi trường ảo
python -m venv motion_env

# Kích hoạt môi trường ảo
# Trên Windows
motion_env\Scripts\activate
# Trên MacOS/Linux
source motion_env/bin/activate

# Cài đặt các thư viện từ file requirements.txt
pip install -r requirements.txt
```
