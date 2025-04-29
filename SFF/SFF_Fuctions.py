import numpy as np
import cv2
from scipy.optimize import curve_fit
from scipy.ndimage import convolve
from typing import List, Tuple

class ModifiedSFF:
    def __init__(self, img_sequence: List[np.ndarray], z_positions: List[float]):
        """
        初始化改进的离焦测深类
        
        参数:
            img_sequence: 图像序列，每个元素是一个二维numpy数组
            z_positions: 每个图像对应的z轴位置
        """
        self.img_sequence = img_sequence
        self.z_positions = np.array(z_positions)
        self.height, self.width = img_sequence[0].shape
        self.num_frames = len(img_sequence)
        
        # 定义GDANS算子使用的卷积核
        self.H1 = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        
        # 定义图像处理算子使用的卷积核
        self.H3 = 0.5 * np.array([[-1, 0, 1]])
        self.H2 = np.array([0.275, 0.45, 0.275])
        
        # 初始化聚焦体积矩阵
        self.focus_volume = np.zeros((self.height, self.width, self.num_frames))
        self.enhanced_focus_volume = np.zeros_like(self.focus_volume)
        
        # 自适应邻域大小参数
        self.wL = 2  # 小邻域半径
        self.wH = 4  # 大邻域半径
        self.TH_nh = 30  # 邻域大小切换阈值
    
    def compute_focus_volume(self):
        """计算初始聚焦体积矩阵"""
        for p in range(self.num_frames):
            img = self.img_sequence[p]
            for y in range(self.height):
                for x in range(self.width):
                    # 计算自适应邻域大小
                    std_dev = np.std([self.img_sequence[k][y,x] for k in range(self.num_frames)])
                    wr = self.wL if std_dev >= self.TH_nh else self.wH
                    
                    # 定义邻域
                    y_min = max(0, y - wr)
                    y_max = min(self.height, y + wr + 1)
                    x_min = max(0, x - wr)
                    x_max = min(self.width, x + wr + 1)
                    neighborhood = img[y_min:y_max, x_min:x_max]
                    
                    # 计算局部均值
                    mu = np.mean(neighborhood)
                    
                    # 应用GDANS算子
                    filtered = convolve(neighborhood, self.H1, mode='constant')
                    focus_value = np.sum((filtered - mu)**2)
                    
                    self.focus_volume[y, x, p] = focus_value
    
    def enhance_focus_volume(self):
        """使用图像处理算子增强聚焦体积矩阵"""
        for p in range(self.num_frames):
            img = self.img_sequence[p]
            for y in range(1, self.height-1):
                for x in range(1, self.width-1):
                    # 计算梯度幅值
                    grad_x = convolve(img[y-1:y+2, x-1:x+2], self.H3, mode='constant')
                    grad_y = convolve(img[y-1:y+2, x-1:x+2], self.H3.T, mode='constant')
                    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                    g = np.sum(grad_mag)
                    
                    # 应用图像处理算子
                    if p == 0:
                        IPO = self.H2[1] * g + self.H2[2] * \
                              np.sum(np.sqrt(convolve(self.img_sequence[p+1][y-1:y+2, x-1:x+2], self.H3)**2 + 
                                     convolve(self.img_sequence[p+1][y-1:y+2, x-1:x+2], self.H3.T)**2)
                    elif p == self.num_frames - 1:
                        IPO = self.H2[0] * \
                              np.sum(np.sqrt(convolve(self.img_sequence[p-1][y-1:y+2, x-1:x+2], self.H3)**2 + 
                                     convolve(self.img_sequence[p-1][y-1:y+2, x-1:x+2], self.H3.T)**2)) + \
                              self.H2[1] * g
                    else:
                        IPO = self.H2[0] * \
                              np.sum(np.sqrt(convolve(self.img_sequence[p-1][y-1:y+2, x-1:x+2], self.H3)**2 + 
                                     convolve(self.img_sequence[p-1][y-1:y+2, x-1:x+2], self.H3.T)**2)) + \
                              self.H2[1] * g + \
                              self.H2[2] * \
                              np.sum(np.sqrt(convolve(self.img_sequence[p+1][y-1:y+2, x-1:x+2], self.H3)**2 + 
                                     convolve(self.img_sequence[p+1][y-1:y+2, x-1:x+2], self.H3.T)**2))
                    
                    # 增强聚焦值
                    self.enhanced_focus_volume[y, x, p] = IPO * self.focus_volume[y, x, p]
    
    def gaussian_fit(self, x, a, b, c):
        """高斯拟合函数"""
        return a * np.exp(-(x - b)**2 / (2 * c**2))
    
    def get_optimal_focus_positions(self):
        """获取每个像素点的最佳聚焦位置"""
        depth_map = np.zeros((self.height, self.width))
        
        for y in range(self.height):
            for x in range(self.width):
                # 获取增强聚焦测度曲线
                efmc = self.enhanced_focus_volume[y, x, :]
                
                # 跳过通孔区域（聚焦值全为0）
                if np.all(efmc == 0):
                    depth_map[y, x] = np.nan
                    continue
                
                # 找到最大聚焦值的位置
                pmax = np.argmax(efmc)
                max_val = efmc[pmax]
                
                # 确定拟合范围
                left = pmax
                while left > 0 and efmc[left] > 0.5 * max_val:
                    left -= 1
                
                right = pmax
                while right < self.num_frames - 1 and efmc[right] > 0.5 * max_val:
                    right += 1
                
                # 确保有足够的数据点进行拟合
                if right - left < 3:
                    depth_map[y, x] = self.z_positions[pmax]
                    continue
                
                # 准备拟合数据
                x_fit = np.arange(left, right + 1)
                y_fit = efmc[left:right + 1]
                
                try:
                    # 初始猜测参数
                    p0 = [max_val, pmax, (right - left)/2]
                    
                    # 执行高斯拟合
                    popt, _ = curve_fit(self.gaussian_fit, x_fit, y_fit, p0=p0)
                    
                    # 获取最佳聚焦位置（高斯峰中心）
                    optimal_p = popt[1]
                    
                    # 确保拟合结果在合理范围内
                    if left <= optimal_p <= right:
                        # 插值得到z位置
                        if optimal_p <= 0:
                            depth_map[y, x] = self.z_positions[0]
                        elif optimal_p >= self.num_frames - 1:
                            depth_map[y, x] = self.z_positions[-1]
                        else:
                            # 线性插值
                            p1 = int(np.floor(optimal_p))
                            p2 = int(np.ceil(optimal_p))
                            alpha = optimal_p - p1
                            depth_map[y, x] = (1 - alpha) * self.z_positions[p1] + alpha * self.z_positions[p2]
                    else:
                        depth_map[y, x] = self.z_positions[pmax]
                except:
                    # 如果拟合失败，使用最大聚焦值位置
                    depth_map[y, x] = self.z_positions[pmax]
        
        return depth_map
    
    def reconstruct_3d(self):
        """执行完整的三维重建流程"""
        print("Computing initial focus volume...")
        self.compute_focus_volume()
        
        print("Enhancing focus volume...")
        self.enhance_focus_volume()
        
        print("Calculating optimal focus positions...")
        depth_map = self.get_optimal_focus_positions()
        
        return depth_map


# 示例使用代码
if __name__ == "__main__":
    # 假设我们已经加载了图像序列和对应的z位置
    # img_sequence = [img1, img2, ..., imgN]  # 每个img是一个灰度图像numpy数组
    # z_positions = [z1, z2, ..., zN]  # 每个图像对应的z轴位置
    
    # 创建改进的SFF实例
    # sff = ModifiedSFF(img_sequence, z_positions)
    
    # 执行三维重建
    # depth_map = sff.reconstruct_3d()
    
    # 可视化深度图
    # import matplotlib.pyplot as plt
    # plt.imshow(depth_map, cmap='jet')
    # plt.colorbar()
    # plt.title("Reconstructed Depth Map")
    # plt.show()
    
    print("This is a template implementation. Please load actual image sequence and z positions to use.")