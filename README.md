# Hand Gesture Detection

## Mô tả dự án

Dự án Hand Gesture Detection là một hệ thống nhận diện cử chỉ tay dựa trên AI, sử dụng PyTorch và dữ liệu từ camera để phát hiện và nhận diện các cử chỉ tay trong thời gian thực. Hệ thống sử dụng mô hình học sâu để học các đặc trưng từ hình ảnh tay (21 điểm đặc trưng) và phân loại chúng thành các nhãn cử chỉ khác nhau.

### Mục tiêu

- Phát hiện các cử chỉ tay phổ biến (mở tay, nắm tay, ngón cái, ký hiệu hòa bình, v.v.).
- Sử dụng dữ liệu từ camera để nhận diện cử chỉ tay thời gian thực.
- Cung cấp mã nguồn đầy đủ và cấu trúc rõ ràng để mở rộng và tích hợp vào các hệ thống khác.

---

## Cấu trúc thư mục dự án

hand_gesture_detection/
├── data/
│ ├── raw/ # Dữ liệu thô từ camera hoặc ảnh tĩnh
│ ├── processed/ # Dữ liệu đã được xử lý dưới dạng CSV
├── models/ # Thư mục chứa mô hình đã huấn luyện và mô hình định nghĩa
│ ├── finger_gesture_model.py
│ └── hand_gesture_model.pth # File mô hình đã được huấn luyện
├── scripts/ # Các script chính để xử lý, huấn luyện và phát hiện
│ ├── preprocess.py # Script xử lý dữ liệu và lưu dưới dạng CSV
│ ├── train.py # Script huấn luyện mô hình từ dữ liệu CSV
│ ├── detect.py # Script phát hiện cử chỉ tay qua camera thời gian thực
│ ├── capture_camera.py # Mới: Chức năng capture video từ camera
│ └── train_from_camera.py # Mới: Huấn luyện mô hình từ dữ liệu camera
├── utils/ # Các hàm tiện ích hỗ trợ xử lý dữ liệu và load mô hình
│ ├── data_loader.py # Hàm load dữ liệu từ file CSV
│ ├── hand_landmarks_extractor.py # Hàm trích xuất 21 điểm đặc trưng từ ảnh
├── requirements.txt # File chứa các thư viện cần thiết để chạy dự án
├── README.md # Mô tả thông tin dự án và hướng dẫn sử dụng

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
