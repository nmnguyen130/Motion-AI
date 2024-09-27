import os
import numpy as np
import pandas as pd
import torch

def load_labels(label_file):
    """
    Hàm để load nhãn từ file CSV.
    Arguments:
    - label_file: Đường dẫn đến file chứa nhãn.

    Returns:
    - labels: Từ điển ánh xạ các nhãn cử chỉ thành số nguyên.
    """
    
    df = pd.read_csv(label_file)
    labels = {row['label']: row['value'] for _, row in df.iterrows()}
    return labels

def load_data(csv_path, labels):
    """
    Hàm để load dữ liệu đã xử lý từ file CSV.
    Arguments:
    - csv_path: Đường dẫn đến file CSV chứa dữ liệu đã xử lý.
    - labels: Từ điển ánh xạ các nhãn cử chỉ thành số nguyên.

    Returns:
    - data: Tensor chứa các điểm đặc trưng của cử chỉ.
    - target: Tensor chứa nhãn tương ứng với dữ liệu.
    """

    data, target = [], []
    
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(csv_path)
    points = df.iloc[:, :-1].values  # Bỏ cột nhãn (cột cuối cùng)
    label_column = df.iloc[:, -1].values  # Cột nhãn

    for i, label in enumerate(label_column):
        if label in labels:
            data.append(points[i])
            target.append(labels[label])

    # Chuyển đổi dữ liệu thành tensor
    data = torch.tensor(np.array(data), dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.long)
    return data, target