# time: 2024/1/27 10:19
# author: YanJP
import numpy as np
import matplotlib.pyplot as plt

# 定义视频数量
num_videos = 10

# 生成Zipf分布的概率
a = 3.5 # Zipf分布的参数，可以调整
zipf_probs = np.random.zipf(a, num_videos)

# 归一化概率，使其总和为1
zipf_probs_normalized = zipf_probs / np.sum(zipf_probs)

# 打印生成的概率
print("生成的Zipf分布概率：", zipf_probs_normalized)

# 绘制直方图
plt.bar(range(1, num_videos + 1), zipf_probs_normalized)
plt.xlabel('视频')
plt.ylabel('概率')
plt.title('Zipf分布概率')
plt.show()

# ones = np.ones(10)
# print(ones)
#
# video_ids = np.array([1, 2, 3, 4, 5])  # 视频ID数组
# probabilities = np.array([0.3, 0.1, 0.6, 0.8, 0.2])  # 对应视频ID的概率数组
#
# # 根据概率值对视频ID进行排序
# sorted_indices = probabilities.argsort()
#
# # 使用索引获取排序后的视频ID
# sorted_video_ids = video_ids[sorted_indices]
#
# print(sorted_video_ids)