from enum import Enum

class ReferenceEdgeType(int, Enum):
    internal = 1  # Match các kiểu tham chiếu tới điều luật ngay đằng trước | Match các kiểu tham chiếu trực tiếp tên điều luật
    external = 2  # Match kiểu tham chiếu tự sinh (các article gần article hiện tại)

class DataFormat(str, Enum):
    ALQAC = "ALQAC"
    COLIEE = "COLIEE"