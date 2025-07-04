import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import open3d as o3d
from scipy.optimize import curve_fit

# 参数设置
wr_L = 2  # 自适应邻域半径 wr 的较小值
wr_H = 8  # 自适应邻域半径 wr 的较大值
TH_mh = 600 # 阈值 TH_mh
w_L = 2  # 参数 wL
w_H = 4  # 参数 wH
delta_z = 10  # 采样间隔 Δz (μm)
total_frames = 97  # 图像帧总数 P


# 读取图像序列
def load_image_sequence(folder_path, total_frames):
    image_sequence = []
    for p in range(1, total_frames + 1):
        img_path = f"{folder_path}/frame_{p:03d}.png"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            image_sequence.append(img)
    return image_sequence



# 计算灰度梯度
def calculate_gray_gradient(image_sequence):
    gradients = []
    H3 = np.array([[-1, 0, 1]])  # H3 算子
    H3 = H3 / 2
    HT3 = H3.T  # H3 的转置
    for img in image_sequence:
        grad_x = cv2.filter2D(img, cv2.CV_64F, H3)
        grad_y = cv2.filter2D(img, cv2.CV_64F, HT3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradients.append(grad_mag)
    return gradients



# 计算焦点测量值
def calculate_focus_measure(image_sequence, varmax):
    height, width = image_sequence[0].shape
    focus_volume = np.zeros((height, width, total_frames))
    for p in range(total_frames):
        img = image_sequence[p]
        for x in range(width):
            for y in range(height):
                if x < wr_L or x >= width - wr_L or y < wr_L or y >= height - wr_L:
                    wr = wr_L
                else:
                    neighborhood = img[y - wr_L:y + wr_L + 1, x - wr_L:x + wr_L + 1]
                    var = np.var(neighborhood)
                    varmax = max(varmax, var)
                    if var >= TH_mh:
                        wr = wr_L
                    else:
                        wr = wr_H
                if wr == wr_L:
                    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                else:
                    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                    # kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
                neighborhood = img[max(y - wr, 0):min(y + wr + 1, height),
                                max(x - wr, 0):min(x + wr + 1, width)]
                filtered = cv2.filter2D(neighborhood, -1, kernel)
                focus_volume[y, x, p] = np.sum(filtered ** 2)
    print("varmax=",varmax)
    return focus_volume

# 图像处理操作增强焦点体积矩阵
def enhance_focus_volume(focus_volume):
    enhanced_volume = np.zeros_like(focus_volume)
    H2 = np.array([0.275, 0.45, 0.275])
    H2 = H2 / np.sum(H2)
    for p in range(total_frames):
        for x in range(focus_volume.shape[1]):
            for y in range(focus_volume.shape[0]):
                start = max(p - 1, 0)
                end = min(p + 1, total_frames - 1)
                neighborhood = focus_volume[y, x, start:end + 1]
                if len(neighborhood) == 3:
                    enhanced_value = np.sum(neighborhood * H2)
                    enhanced_volume[y, x, p] = enhanced_value
                else:
                    enhanced_volume[y, x, p] = focus_volume[y, x, p]
    return enhanced_volume

"""
# 改进的高斯曲线拟合方法
def modified_gaussian_fit(focus_curve, p_max, P_L, P_R, TH_c=0.5):
    zp = np.arange(len(focus_curve)) * delta_z
    # 初始化参数
    sigma = 1.0
    # 定义高斯曲线模型
    def gaussian_model(zp, sigma):
        return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((zp - p_max * delta_z) / sigma) ** 2)
    # 使用曲线拟合
    try:
        popt, _ = curve_fit(gaussian_model, zp[P_L:P_R+1], focus_curve[P_L:P_R+1], p0=[sigma])
        sigma_opt = popt[0]
    except:
        sigma_opt = sigma
    # 计算最佳焦点位置
    z_opt = p_max * delta_z
    return z_opt
"""




# 改进的高斯曲线拟合方法
def modified_gaussian_fit(focus_curve, p_max, delta_z, TH_c=0.5):
    zp = np.arange(len(focus_curve)) * delta_z
    # 初始化参数
    sigma = 1.0
    # 定义高斯曲线模型
    def gaussian_model(zp, sigma):
        return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((zp - p_max * delta_z) / sigma) ** 2)
    # 计算ΔP_L和ΔP_R
    delta_P_L = p_max - (p_max - 2 * TH_c)
    delta_P_R = (p_max + 2 * TH_c) - p_max
    P_L = int(max(0, p_max - delta_P_L))
    P_R = int(min(len(focus_curve), p_max + delta_P_R))
    # 使用曲线拟合
    try:
        popt, _ = curve_fit(gaussian_model, zp[P_L:P_R+1], focus_curve[P_L:P_R+1], p0=[sigma])
        sigma_opt = popt[0]
    except:
        sigma_opt = sigma
    # 计算最佳焦点位置
    z_opt = p_max * delta_z
    return z_opt

# 获取最佳焦点位置
def get_optimal_focus_position(enhanced_volume):
    optimal_positions = np.zeros((height, width))
    for x in range(width):
        for y in range(height):
            focus_curve = enhanced_volume[y, x, :]
            max_idx = np.argmax(focus_curve)
            # 确定拟合范围
            P_L = max(max_idx - 2, 0)
            P_R = min(max_idx + 2, total_frames - 1)
            # 改进的高斯曲线拟合
            z_opt = modified_gaussian_fit(focus_curve, max_idx, P_L, P_R)
            optimal_positions[y, x] = z_opt
    return optimal_positions


# 绘制灰度梯度图像
def plot_gray_gradients(gray_gradients, frame_indices):
    plt.figure(figsize=(12, 8))
    for idx, p in enumerate(frame_indices):
        plt.subplot(2, 4, idx + 1)
        plt.imshow(gray_gradients[p], cmap='jet')
        plt.colorbar(label='Gradient Magnitude')
        plt.title(f'Gradient of Image Frame {p+1}')
        plt.xlabel('X')
        plt.ylabel('Y')
    plt.tight_layout()
    plt.show()

def plot_focus_curves(focus_volume, enhanced_volume, image_points):
    plt.figure(figsize=(18, 10))
    for idx, (x, y) in enumerate(image_points):
        plt.subplot(2, 3, idx + 1)
        
        # 获取原始和增强的焦点曲线
        focus_curve = focus_volume[y, x, :]
        enhanced_curve = enhanced_volume[y, x, :]
        
        # 归一化（可选）
        focus_curve_norm = (focus_curve - np.min(focus_curve)) / (np.max(focus_curve) - np.min(focus_curve))
        enhanced_curve_norm = (enhanced_curve - np.min(enhanced_curve)) / (np.max(enhanced_curve) - np.min(enhanced_curve))
        
        # 绘制两条曲线，使用不同颜色和线型
        plt.plot(focus_curve_norm, 
                label='Original Focus', 
                color='blue', 
                linestyle='-', 
                alpha=0.7)
        plt.plot(enhanced_curve_norm, 
                label='Enhanced Focus', 
                color='red', 
                linestyle='--', 
                alpha=0.7)
        
        plt.xlabel('Image Frame Index')
        plt.ylabel('Normalized Focus Value')
        plt.title(f'Focus Curves at Point ({x}, {y})')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# 三维重建
def reconstruct_3d(optimal_positions):
    # 根据最佳焦点位置构建 3D 点云
    height_map = optimal_positions
    return height_map

# 生成彩色深度图
def generate_colored_depth_map(height_map):
    plt.figure(figsize=(10, 8))
    norm = Normalize(vmin=np.min(height_map), vmax=np.max(height_map))
    cmap = plt.cm.jet
    colored_depth_map = cmap(norm(height_map))
    plt.imshow(colored_depth_map)
    plt.colorbar(label='Depth (μm)')
    plt.title('Colored Depth Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('colored_depth_map.png')
    plt.show()

# 生成彩色三维点云图
def generate_colored_point_cloud(height_map, image_sequence):
    height, width = height_map.shape
    points = []
    colors = []
    for x in range(width):
        for y in range(height):
            z = height_map[y, x]
            if y < image_sequence[0].shape[0] and x < image_sequence[0].shape[1]:
                r = image_sequence[0][y, x]  # 使用灰度值作为颜色信息
                g = image_sequence[0][y, x]
                b = image_sequence[0][y, x]
            else:
                r = g = b = 0
            points.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])
    return np.array(points), np.array(colors)

# 生成三维重建图
def generate_3d_surface_plot(height_map):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(height_map.shape[1])
    y = np.arange(height_map.shape[0])
    x, y = np.meshgrid(x, y)
    z = height_map

    surf = ax.plot_surface(x, y, z, cmap='jet', edgecolor='none')
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Depth (μm)')
    ax.set_title('3D Reconstructed Surface')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth (μm)')
    plt.savefig('3d_reconstructed_surface.png')
    plt.show()

# 主函数
if __name__ == "__main__":
    t1 = time.time()
    
    folder_path = r"C:\Users\Administrator\Desktop\picture\class4"
    image_sequence = load_image_sequence(folder_path, total_frames)

    t2 = time.time()
    print("读入图片用时（秒）：",t2 - t1)
    
    if not image_sequence:
        print("图像序列为空，请检查文件路径和文件名格式。")
        exit()
    height, width = image_sequence[0].shape

    t3 = time.time()
    
    gray_gradients = calculate_gray_gradient(image_sequence)

    t4 = time.time()

    print("计算灰度梯度用时（秒）：",t4 - t3)
    varmax = 0
    focus_volume = calculate_focus_measure(image_sequence, varmax)

    t5 = time.time()

    print("计算焦点测量值用时（秒）：",t5 - t4)

    enhanced_volume = enhance_focus_volume(focus_volume)
    #enhanced_volume = enhance_focus_volume(focus_volume)

    t6 = time.time()

    print("计算增强焦点体积矩阵用时（秒）：",t6 - t5)

    frame_indices = [0, 20, 40, 60, 80, 96]

    plot_gray_gradients(gray_gradients, frame_indices)

    t7 = time.time()

    # print("输出",5,"张灰度梯度图用时（秒）：",t7 - t6)

    image_points = [(20, 20), (60, 150), (136, 239), (131, 156), (214, 239), (30, 250)]
    plot_focus_curves(focus_volume, enhanced_volume, image_points)

    t8= time.time()
    print("输出",8,"个点的聚焦值变化曲线用时（秒）：", t8 - t7)

    # 获取最佳焦点位置
    optimal_positions = get_optimal_focus_position(enhanced_volume)
    # 三维重建
    height_map = reconstruct_3d(optimal_positions)
    # 生成彩色深度图
    generate_colored_depth_map(height_map)
    # 生成三维重建图
    generate_3d_surface_plot(height_map)
    # 生成彩色三维点云图
    points, colors = generate_colored_point_cloud(height_map, image_sequence)
    # 使用 Open3D 显示三维点云图
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    
    print("程序总用时（秒）：",t8 - t1)
    
