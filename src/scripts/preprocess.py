import os
import pandas as pd
from utils.hand_utils import extract_landmarks

# Đường dẫn dữ liệu
data_dir = '../data/raw'
processed_dir = '../data/processed'

# Tạo thư mục chứa dữ liệu đã xử lý nếu chưa có
os.makedirs(processed_dir, exist_ok=True)

# Xử lý từng ảnh trong thư mục raw
for filename in os.listdir(data_dir):
    img_path = os.path.join(data_dir, filename)
    points = extract_landmarks(img_path)
    
    # Lưu các điểm đặc trưng vào file CSV nếu phát hiện tay trong ảnh
    if points is not None:
        # Tạo một DataFrame với tên cột tương ứng
        df = pd.DataFrame([points], columns=[f'point_{i}' for i in range(len(points))])
        csv_filename = filename.replace('.jpg', '.csv')
        df.to_csv(os.path.join(processed_dir, csv_filename), index=False)
