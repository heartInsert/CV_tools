import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# 输入coco形式的 json 文件
instance_json_path = '/home/xjz/Desktop/Coding/PycharmProjects/competition/Help_Tools/CV_tools/data/DcicCoco_train.json'
with open(instance_json_path, 'r') as f:
    json_file = json.load(f)
annotations = json_file['annotations']
width_list = []
height_list = []
width_height_ratio = []
for row in annotations:
    x_min, y_min, width, height = row['bbox'].split(",")
    width_list.append(width)
    height_list.append(height)
    width_height_ratio.append(width / height)
plt.hist(width_list, bins=10, color='b', alpha=0.3)
plt.hist(height_list, bins=10, color='b', alpha=0.3)
plt.hist(width_height_ratio, bins=10, color='b', alpha=0.3)
plt.show()
print()
