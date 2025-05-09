import cv2
import os
import numpy as np
from datetime import datetime
import time
import glob

crop_size = 500

# 打开默认摄像头
cap = cv2.VideoCapture(0)

def draw_crosshair(frame):
    """在帧的中心绘制十字线（不修改原帧，返回新帧）"""
    # 复制原帧，避免修改原始数据
    frame_with_crosshair = frame.copy()
    frame_with_crosshair = cv2.resize(frame, (1280, 720))
    h, w = frame_with_crosshair.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # 绘制水平线（红色，线宽2px）
    cv2.line(frame_with_crosshair, (0, center_y), (w, center_y), (0, 0, 255), 1)
    # 绘制垂直线（红色，线宽2px）
    cv2.line(frame_with_crosshair, (center_x, 0), (center_x, h), (0, 0, 255), 1)
    
    return frame_with_crosshair

def process_images(input_dir="photo", output_dir="processed", Vision_adjust=50):
    """处理所有图片：灰度化、均值滤波、边缘检测、统一裁切"""
    img_paths = glob.glob(f"{input_dir}/**/original/*.png")
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
        '''保存处理后的图片'''
        # 替换路径中的original为processed
        save_path_gray = os.path.join(os.path.dirname(img_path).replace("original", "gray"), os.path.basename(img_path))
        os.makedirs(os.path.dirname(save_path_gray), exist_ok=True)
        cv2.imwrite(save_path_gray, gray)

        blurred = cv2.blur(gray, (5, 5))
        edges = cv2.Canny(blurred, 50, 150)

        # 将边缘部分的像素值设置为红色
        blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        blurred_bgr[edges == 255] = [0, 0, 255]
        '''保存处理后的图片'''
        # 替换路径中的original为processed
        save_path_blur = os.path.join(os.path.dirname(img_path).replace("original", "edges"), os.path.basename(img_path))
        os.makedirs(os.path.dirname(save_path_blur), exist_ok=True)
        cv2.imwrite(save_path_blur, blurred_bgr)
        
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
    if not standard_edges:
        print("未检测到有效边缘！")
        return

    min_x1, min_y1, max_x2, max_y2 = standard_edges
    print(f"使用最大边缘范围：({min_x1}, {min_y1}) - ({max_x2}, {max_y2})")

    # 第二步：按最大边缘范围裁切所有图片（处理灰度图）
    save_path_gray = os.path.dirname(save_path_gray)
    gray_path = glob.glob(f"{save_path_gray}/*.png")
    for img_path in gray_path:
        img = cv2.imread(img_path)
        if img is None:
            continue

        # # 裁切灰度图
        img = img[min_y1:max_y2, min_x1:max_x2]

        '''保存处理后的图片'''
        # 替换路径中的gray为processed
        save_path = os.path.join(os.path.dirname(img_path).replace("gray", "processed"), os.path.basename(img_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)

    print(f"所有图片处理完成，保存至 {output_dir} 目录！")