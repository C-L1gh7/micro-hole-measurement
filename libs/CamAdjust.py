import cv2
import os
import numpy as np
from datetime import datetime
import time

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

def CamAdjust(cap):
    while 1:
        ret, frame = cap.read()
        # 在预览画面添加十字线
        frame_with_crosshair = draw_crosshair(frame) #复制当前帧，缩放为720p并添加十字线
        cv2.imshow('Camera (Press "q" to quit)', frame_with_crosshair)
        key = cv2.waitKey(1) & 0xFF
        if key ==  ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()