#加载图像序列
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 假设图像存储在文件夹中，按顺序命名(如frame_001.png, frame_002.png等)
image_folder = "path/to/your/images"
num_images = 50  # 你的图像数量
img_sequence = []

for i in range(1, num_images+1):
    # 读取图像并转换为灰度
    img = cv2.imread(f"{image_folder}/frame_{i:03d}.png", cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img_sequence.append(img)
    else:
        print(f"无法加载图像: frame_{i:03d}.png")

# 检查图像尺寸是否一致
for i in range(1, len(img_sequence)):
    if img_sequence[i].shape != img_sequence[0].shape:
        print(f"图像尺寸不一致: 图像0 {img_sequence[0].shape}, 图像{i} {img_sequence[i].shape}")

#准备z位置数据
# 假设z位置是等间隔的，从0开始，每次移动10微米
z_step = 10  # 微米
z_positions = [i * z_step for i in range(len(img_sequence))]

# 或者从文件中读取实际z位置
# z_positions = np.loadtxt("z_positions.txt")
#初始化并运行SFF算法
from modified_sff import ModifiedSFF  # 假设代码保存在modified_sff.py中

# 创建SFF实例
sff = ModifiedSFF(img_sequence, z_positions)

# 执行三维重建
depth_map = sff.reconstruct_3d()

#可视化结果
# 显示深度图
plt.figure(figsize=(10, 8))
plt.imshow(depth_map, cmap='jet')
plt.colorbar(label='Depth (μm)')
plt.title("Reconstructed Depth Map")
plt.show()

# 显示3D表面
from mpl_toolkits.mplot3d import Axes3D

# 创建网格
x = np.arange(depth_map.shape[1]))
y = np.arange(depth_map.shape[0]))
x, y = np.meshgrid(x, y)

# 绘制3D图
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, depth_map, cmap='viridis')

ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
ax.set_zlabel('Depth (μm)')
ax.set_title('3D Reconstruction of Micro-hole Array')
plt.show()
