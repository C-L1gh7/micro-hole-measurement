import cv2
import os
from datetime import datetime
import time
import serial

from libs import shot
from libs import CamAdjust as ca


crop_size = 1000  # 预裁切大小
patch_size = 50  # patch大小

'''配置USART1的参数'''
SERIAL_PORT = 'COM5'
BAUD_RATE = 115200
TIMEOUT = 1

# 打开默认摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 生成文件夹名(年月日_时分)
folder_name = datetime.now().strftime("photo/%Y%m%d_%H%M")
original_dir = os.path.join(folder_name, "original")
processed_dir = os.path.join(folder_name, "processed")

# 如果文件夹不存在则创建
os.makedirs(original_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

try:
    # 初始化串口
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    print(f"已连接串口: {ser.name}")
    
    photo_count = 0
    max_photos = 1000      # 照片数量
    capture_mode = False  # 拍摄模式标志

    cv2.namedWindow('Camera (Press "q" to quit, "s" to start/stop capture)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera (Press "q" to quit, "s" to start/stop capture)', 1280, 720)  # 显示窗口保持16:9比例

    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            break
        
        # 在预览画面添加十字线
        frame_with_crosshair = shot.draw_crosshair(frame, crop_size=crop_size, patch_size=patch_size)
        cv2.imshow('Camera (Press "q" to quit, "s" to start/stop capture)', frame_with_crosshair)
        
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # 调试时允许使用按键控制电机移动
        elif key ==ord('1'): # 前进
            ser.write(b'1')
        elif key ==ord('2'): # 后退
            ser.write(b'2')
        elif key ==ord('3'): # 大角度前进
            ser.write(b'3')
        elif key ==ord('4'): # 大角度后退
            ser.write(b'4')    

        elif key == ord('s'):  # 切换拍摄模式
            capture_mode = not capture_mode
            print(f"拍摄模式: {'开启' if capture_mode else '关闭'}")
            if capture_mode:
                ser.write(b'1')
                print("已发送正转信号 '1'，开始拍摄")
            
        
        
        if capture_mode and photo_count < max_photos:
            # 检查串口数据
            if ser.in_waiting > 0:
                received_data = ser.read(1).decode('ascii')
                
                if received_data == 'D':
                    print(f"收到指令 'D'，拍摄照片 {photo_count+1}/{max_photos}")
                    start_time = time.time()
                    
                    # 裁剪中心区域
                    height, width = frame.shape[:2]
                    center_x, center_y = width // 2, height // 2
                    x1 = max(0, center_x - crop_size // 2)
                    y1 = max(0, center_y - crop_size // 2)
                    x2 = min(width, center_x + crop_size // 2)
                    y2 = min(height, center_y + crop_size // 2)
                    center_crop = frame[y1:y2, x1:x2]
                    
                    # 保存原始帧
                    filename = f"{original_dir}/{photo_count+1}.png"
                    cv2.imwrite(filename, center_crop)
                    
                    # 计算耗时
                    elapsed_time = (time.time() - start_time) * 1000
                    print(f"照片已保存为 {filename}，耗时 {elapsed_time:.2f}ms")
                    
                    # 更新计数器
                    photo_count += 1
                    
                    # 发送正转信号
                    if photo_count < max_photos:
                        ser.write(b'1')
                        print("已发送正转信号 '1'")
                        # time.sleep(2)
                    else:
                        print(f"已完成 {max_photos} 张照片拍摄")
                        capture_mode = False
    
except serial.SerialException as e:
    print(f"串口错误: {e}")
except Exception as e:
    print(f"发生错误: {e}")
finally:
    # 释放资源
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("串口已关闭")
    
    cap.release()
    cv2.destroyAllWindows()
    print("摄像头已释放")

# 处理所有图片
if photo_count > 0:
    print("开始处理图片...")
    shot.process_images(crop_size=crop_size, patch_size=patch_size)
    print("图片处理完成")