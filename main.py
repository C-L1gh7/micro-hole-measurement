import cv2
import os
from datetime import datetime
import time
import serial

from libs import shot
from libs import CamAdjust as ca

start_all = time.time()# 记录开始时间

crop_size = 750  # 预裁切大小
patch_size = 100  # patch大小

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
    save_photos = True    # 照片保存开关，按k键控制
    k_pressed_photo_number = None  # 第一次按下k时的照片编号
    
    # 坐标系统变量
    current_position = 0  # 当前位置坐标
    zero_position = 0     # 零点位置（按s时的位置）
    position_initialized = False  # 是否已初始化坐标系统

    cv2.namedWindow('Camera (Press "q" to quit, "s" to start/stop capture, "r" to reset, "k" to toggle save)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera (Press "q" to quit, "s" to start/stop capture, "r" to reset, "k" to toggle save)', 1280, 720)  # 显示窗口保持16:9比例

    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            break
        
        # 在预览画面添加十字线
        frame_with_crosshair = shot.draw_crosshair(frame, crop_size=crop_size, patch_size=patch_size)
        
        # 在画面上显示当前状态信息
        status_text = f"Position: {current_position - zero_position if position_initialized else 'Not initialized'}"
        status_text += f" | Capture: {'ON' if capture_mode else 'OFF'}"
        status_text += f" | Save: {'ON' if save_photos else 'OFF'}"
        status_text += f" | Photos: {photo_count}/{max_photos}"
        if k_pressed_photo_number is not None:
            status_text += f" | K-pressed at: {k_pressed_photo_number}"
        
        cv2.putText(frame_with_crosshair, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Camera (Press "q" to quit, "s" to start/stop capture, "r" to reset, "k" to toggle save)', frame_with_crosshair)
        
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # 调试时允许使用按键控制电机移动
        elif key == ord('1'): # 前进
            ser.write(b'1')
            current_position += 1
            print(f"手动前进，当前位置: {current_position}, 相对坐标: {current_position - zero_position if position_initialized else 'Not initialized'}")
        elif key == ord('2'): # 后退
            ser.write(b'2')
            current_position -= 1
            print(f"手动后退，当前位置: {current_position}, 相对坐标: {current_position - zero_position if position_initialized else 'Not initialized'}")
        elif key == ord('3'): # 大角度前进（步长10）
            ser.write(b'3')
            current_position += 10
            print(f"大角度前进，当前位置: {current_position}, 相对坐标: {current_position - zero_position if position_initialized else 'Not initialized'}")
        elif key == ord('4'): # 大角度后退（步长10）
            ser.write(b'4')
            current_position -= 10
            print(f"大角度后退，当前位置: {current_position}, 相对坐标: {current_position - zero_position if position_initialized else 'Not initialized'}")
        elif key == ord('r') or key == ord('R'):  # 复位到零点
            if position_initialized:
                distance_to_zero = current_position - zero_position
                print(f"复位中...需要移动距离: {distance_to_zero}")
                
                # 根据距离方向选择移动指令
                if distance_to_zero > 0:
                    # 需要后退
                    for _ in range(abs(distance_to_zero)):
                        ser.write(b'2')
                        time.sleep(0.1)  # 短暂延迟确保指令执行
                elif distance_to_zero < 0:
                    # 需要前进
                    for _ in range(abs(distance_to_zero)):
                        ser.write(b'1')
                        time.sleep(0.1)  # 短暂延迟确保指令执行
                
                current_position = zero_position
                print(f"复位完成，当前位置: {current_position}, 相对坐标: 0")
            else:
                print("坐标系统未初始化，请先按's'开始拍摄模式")
        elif key == ord('k') or key == ord('K'):  # 切换照片保存开关
            save_photos = not save_photos
            # 第一次按下k键时记录照片编号
            if k_pressed_photo_number is None and not save_photos:
                k_pressed_photo_number = photo_count
                print(f"第一次按下k键，记录照片编号: {k_pressed_photo_number}")
            print(f"照片保存: {'开启' if save_photos else '关闭'}")
        elif key == ord('s') or key == ord('S'):  # 切换拍摄模式
            capture_mode = not capture_mode
            print(f"拍摄模式: {'开启' if capture_mode else '关闭'}")
            if capture_mode:
                # 初始化坐标系统
                if not position_initialized:
                    zero_position = current_position
                    position_initialized = True
                    print(f"坐标系统已初始化，零点位置: {zero_position}")
                
                ser.write(b'1')
                current_position += 1
                print(f"已发送正转信号 '1'，开始拍摄，当前位置: {current_position}, 相对坐标: {current_position - zero_position}")
            
        
        
        if capture_mode and photo_count < max_photos:
            # 检查串口数据
            if ser.in_waiting > 0:
                received_data = ser.read(1).decode('ascii')
                
                if received_data == 'D':
                    coordinate = current_position - zero_position
                    print(f"收到指令 'D'，位置坐标: {coordinate}")
                    
                    if save_photos:
                        print(f"拍摄照片，坐标: {coordinate}")
                        start_time = time.time()
                        
                        # 裁剪中心区域
                        height, width = frame.shape[:2]
                        center_x, center_y = width // 2, height // 2
                        x1 = max(0, center_x - crop_size // 2)
                        y1 = max(0, center_y - crop_size // 2)
                        x2 = min(width, center_x + crop_size // 2)
                        y2 = min(height, center_y + crop_size // 2)
                        center_crop = frame[y1:y2, x1:x2]
                        
                        # 保存原始帧，文件名为坐标值
                        filename = f"{original_dir}/{coordinate}.png"
                        cv2.imwrite(filename, center_crop)
                        
                        # 计算耗时
                        elapsed_time = (time.time() - start_time) * 1000
                        print(f"照片已保存为 {filename}，耗时 {elapsed_time:.2f}ms")
                        
                        # 更新计数器
                        photo_count += 1
                    else:
                        print(f"位置坐标: {coordinate}，照片保存已关闭，跳过保存")
                    
                    # 发送正转信号
                    if photo_count < max_photos:
                        ser.write(b'1')
                        current_position += 1
                        print(f"已发送正转信号 '1'，当前位置: {current_position}, 相对坐标: {current_position - zero_position}")
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
    # 将k_pressed_photo_number传递给处理函数
    shot.process_images(crop_size=crop_size, patch_size=patch_size, k_pressed_photo_number=k_pressed_photo_number)
    print("图片处理完成")

from libs.focus_analysis import analyze_focus

# 一键分析，显示详细信息并保存结果
result = analyze_focus(adjust=0.053195)

# 获取最佳图片文件名
if result['top_peak']:
    print(f"TOP最佳: {result['top_peak']}")
if result['bottom_peak']:
    print(f"BOTTOM最佳: {result['bottom_peak']}")
if result['aperture']:
    print(f"孔深: {result['aperture']:.5f}")

# 计算总耗时
end_all = time.time()
print(f"总耗时: {(end_all - start_all):.2f} 秒")