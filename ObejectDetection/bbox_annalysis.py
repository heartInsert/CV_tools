import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

bbox = pd.read_csv('train.csv', encoding='gbk')
width_list = []
height_list = []
width_height_ratio = []
for index, row in bbox.iterrows():
    bboxes = row['坐标'].split(";")
    for bbox in bboxes:
        x_mid, y_mid, width, height = bbox.split(' ')
        width_list.append(float(width) * 256)
        height_list.append(float(height) * 256)
        width_height_ratio.append(float(width) * 256 / (float(height) * 256))
        pass
    pass
plt.hist(width_list, bins=10, color='b', alpha=0.3)
# plt.hist(height_list, bins=10, color='b', alpha=0.3)
# plt.hist(width_height_ratio, bins=10, color='b', alpha=0.3)
plt.show()
print()
