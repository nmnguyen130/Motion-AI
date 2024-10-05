import pandas as pd

class LabelHandler:
    def __init__(self, label_file):
        self.labels = self.load_labels(label_file)
        self.reverse_labels = {value: key for key, value in self.labels.items()}

    def load_labels(self, label_file):
        """
        Đọc file CSV và tải label vào dictionary.
        """
        df = pd.read_csv(label_file)
        return {row['label']: row['value'] for _, row in df.iterrows()}

    def get_label_by_index(self, index):
        """
        Lấy nhãn dựa vào chỉ số.
        """
        return self.reverse_labels.get(index, None)

    def get_index_by_label(self, label):
        """
        Lấy chỉ số dựa vào tên nhãn.
        """
        return self.labels.get(label, None)