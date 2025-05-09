import cv2
import os
import numpy as np
from datetime import datetime
import time
import glob

Vision_adjust = 50
crop_size = 500

# 打开默认摄像头
cap = cv2.VideoCapture(0)

def draw_crosshair(frame):
    """在帧的中心绘制十字线（不修改原帧，返回新帧）"""
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # 复制原帧，避免修改原始数据
    frame_with_crosshair = frame.copy()
    
    # 绘制水平线（红色，线宽2px）
    cv2.line(frame_with_crosshair, (0, center_y), (w, center_y), (0, 0, 255), 2)
    # 绘制垂直线（红色，线宽2px）
    cv2.line(frame_with_crosshair, (center_x, 0), (center_x, h), (0, 0, 255), 2)
    
    return frame_with_crosshair
def find_max_edge_image(img_paths):
    """找到边缘最丰富的图片（白色像素点最多）"""
    max_white = 0
    standard_edges = None
    
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.blur(gray, (5, 5))
        edges = cv2.Canny(blurred, 50, 150)
        
        # 计算白色像素点数量
        white_pixels = cv2.countNonZero(edges)
        
        if white_pixels > max_white:
            max_white = white_pixels
            # 找到该图片的边缘范围
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                all_points = np.vstack(contours)
                x, y, w, h = cv2.boundingRect(all_points)
                standard_edges = (x-Vision_adjust, y-Vision_adjust, x+w+Vision_adjust, y+h+Vision_adjust)
    
    return standard_edges

def process_images(input_dir="photo", output_dir="processed"):
    """处理所有图片：灰度化、均值滤波、边缘检测、统一裁切"""
    img_paths = glob.glob(f"{input_dir}/**/original/*.png", recursive=True)
    if not img_paths:
        print(f"未找到 {input_dir} 目录下的图片！")
        return
    
    """找到边缘最丰富的图片（白色像素点最多）"""
    max_white = 0
    standard_edges = None
    
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.blur(gray, (5, 5))
        edges = cv2.Canny(blurred, 50, 150)
        
        # 计算白色像素点数量
        white_pixels = cv2.countNonZero(edges)
        
        if white_pixels > max_white:
            max_white = white_pixels
            # 找到该图片的边缘范围
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                all_points = np.vstack(contours)
                x, y, w, h = cv2.boundingRect(all_points)
                standard_edges = (x-Vision_adjust, y-Vision_adjust, x+w+Vision_adjust, y+h+Vision_adjust)

    # 第一步：找到边缘最丰富的图片作为标准
    standard_edges = find_max_edge_image(img_paths)
    if not standard_edges:
        print("未检测到有效边缘！")
        return

    min_x1, min_y1, max_x2, max_y2 = standard_edges
    print(f"使用最大边缘范围：({min_x1}, {min_y1}) - ({max_x2}, {max_y2})")

    # 第二步：按最大边缘范围裁切所有图片（处理灰度图）
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        
        # # 裁切处理后的灰度图
        gray = gray[min_y1:max_y2, min_x1:max_x2]

        # 保存处理后的图片
        rel_path = os.path.relpath(img_path, input_dir)
        # 替换路径中的original为processed
        rel_path = rel_path.replace("original", "processed")
        save_path = os.path.join(os.path.dirname(img_path).replace("original", "processed"), os.path.basename(img_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, gray)

    print(f"所有图片处理完成，保存至 {output_dir} 目录！")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头！")
        break

    # 在预览画面添加十字线
    frame_with_crosshair = draw_crosshair(frame)
    cv2.imshow('Camera (Press "k" to take photo, "q" to quit)', frame_with_crosshair)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('k'):
        start_time = time.time()  # 记录开始时间

        # 获取图像尺寸
        height, width = frame.shape[:2]
        
        # 计算中心500x500区域的坐标
        center_x, center_y = width // 2, height // 2
        x1 = center_x - crop_size // 2
        y1 = center_y - crop_size // 2
        x2 = center_x + crop_size // 2
        y2 = center_y + crop_size // 2
        
        # 确保裁剪区域不超出图像边界
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # 裁剪中心区域
        center_crop = frame[y1:y2, x1:x2]
        
        # 生成文件夹名(年月日_时分)
        folder_name = datetime.now().strftime("photo/%Y%m%d_%H%M")
        original_dir = os.path.join(folder_name, "original")
        processed_dir = os.path.join(folder_name, "processed")
        
        # 如果文件夹不存在则创建
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        
        # 查找已存在的照片数量以确定新照片编号
        existing_photos = [f for f in os.listdir(original_dir) if f.endswith('.png')]
        photo_number = len(existing_photos) + 1
        
        # 生成文件名
        filename = f"{original_dir}/{photo_number}.png"
        
        # 保存原始帧（不含十字线）
        cv2.imwrite(filename, center_crop)
        
        # 计算耗时
        elapsed_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        print(f"照片已保存为 {filename}，耗时 {elapsed_time:.2f}ms")

# 退出摄像头后处理所有图片
cap.release()
cv2.destroyAllWindows()
process_images()