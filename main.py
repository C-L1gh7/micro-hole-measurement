import cv2
import os
from datetime import datetime
import time
import serial

from libs import shot
from libs import CamAdjust as ca

crop_size = 1000 #预裁切大小

'''配置USART1的参数'''
SERIAL_PORT = 'COM5'
BAUD_RATE = 115200
TIMEOUT = 1

# 打开默认摄像头
cap = cv2.VideoCapture(0)

# 生成文件夹名(年月日_时分)
folder_name = datetime.now().strftime("photo/%Y%m%d_%H%M")
original_dir = os.path.join(folder_name, "original")
processed_dir = os.path.join(folder_name, "processed")

# 如果文件夹不存在则创建
os.makedirs(original_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

#################

####testjzk

i = 1
try:
    # 初始化串口
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    print(f"已连接串口: {ser.name}")

    while 1:
        ret, frame = cap.read()
        # 在预览画面添加十字线
        frame_with_crosshair = shot.draw_crosshair(frame) #复制当前帧，缩放为720p并添加十字线
        cv2.imshow('Camera (Press "q" to quit)', frame_with_crosshair)
        key = cv2.waitKey(1) & 0xFF
        if key ==  ord('q'):
            break
        elif key ==ord('f'):
            ser.write(b'F')
        elif key ==ord('r'):
            ser.write(b'R')
    
    ser.write(b'F')
    print("已发送正转信号 'F'")
    while i<=300:
        i += 1
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头！")
            break

        # 在预览画面添加十字线
        frame_with_crosshair = shot.draw_crosshair(frame) #复制当前帧，缩放为720p并添加十字线
        cv2.imshow('Camera (Press "q" to quit)', frame_with_crosshair)

        if ser.in_waiting > 0:
            received_data = ser.read(1).decode('ascii')  # 读取1个字节
            
            if received_data == 'D':
                print("收到指令 'D'，开始执行...")

                start_time = time.time()  # 记录开始时间
                time.sleep(0.05) #50ms消抖

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

                # 发送正转信号
                ser.write(b'F')
                print("已发送正转信号 'F'")
                
except serial.SerialException as e:
    print(f"串口错误: {e}")
except KeyboardInterrupt:
        print("程序终止")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("串口已关闭")

# 退出摄像头后处理所有图片
cap.release()
cv2.destroyAllWindows()
shot.process_images(Vision_adjust = 50)