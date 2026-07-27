import torch
from torch_cluster import radius_graph
print(radius_graph.__file__)  # 看路径里有没有 "cuda" 或 "cpu"

# 安装torch2.8版本才行，和2.11
# kaolin torch-cluster
# flashatten也装上