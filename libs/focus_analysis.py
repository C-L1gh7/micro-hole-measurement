import cv2
import os
import numpy as np
import pywt
import requests
import hashlib
import base64
from scipy.optimize import curve_fit
from scipy.stats import norm
import glob
import re
from datetime import datetime

def skewed_gaussian(x, a, mu, sigma, alpha):
    """偏态高斯函数"""
    t = (x - mu) / sigma
    return a * 2 * norm.pdf(t) * norm.cdf(alpha * t)

def wavelet_focus_measure(image):
    """小波法计算聚焦度"""
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 3层小波分解
    coeffs = pywt.wavedec2(gray, wavelet='db2', level=3)
    
    # 计算高频系数的能量
    focus_measure = 0.0
    for i in range(1, len(coeffs)):
        cH, cV, cD = coeffs[i]
        focus_measure += np.sum(cH**2) + np.sum(cV**2) + np.sum(cD**2)
    
    return focus_measure / gray.size

def analyze_folder_focus(folder_path):
    """分析文件夹中图像的聚焦度并进行高斯拟合"""
    if not os.path.exists(folder_path):
        return None
    
    # 获取所有图像文件并按数字排序
    image_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if len(image_files) < 3:
        return None
    
    # 提取文件名中的数字并排序
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    image_files.sort(key=extract_number)
    
    # 计算聚焦度
    indices = []
    focus_measures = []
    
    for filename in image_files:
        try:
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
                
            focus_measure = wavelet_focus_measure(image)
            index = extract_number(filename)
            
            indices.append(index)
            focus_measures.append(focus_measure)
            
        except Exception:
            continue
    
    if len(indices) < 3:
        return None
    
    # 转换为numpy数组
    x_data = np.array(indices)
    y_data = np.array(focus_measures)
    
    try:
        # 初始参数估计
        max_focus = np.max(y_data)
        max_index = indices[np.argmax(y_data)]
        sigma_estimate = len(indices) / 6
        initial_guess = [max_focus, max_index, sigma_estimate, 0.0]
        
        # 偏态高斯拟合
        popt, pcov = curve_fit(skewed_gaussian, x_data, y_data, p0=initial_guess)
        
        # 计算真实峰值位置
        x_smooth = np.linspace(min(x_data), max(x_data), 1000)
        y_smooth = skewed_gaussian(x_smooth, *popt)
        peak_index = x_smooth[np.argmax(y_smooth)]
        
        # 计算拟合统计量
        y_pred = skewed_gaussian(x_data, *popt)
        ss_res = np.sum((y_data - y_pred) ** 2)  # 残差平方和
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)  # 总平方和
        r_squared = 1 - (ss_res / ss_tot)  # R²
        
        # 计算RMSE
        rmse = np.sqrt(np.mean((y_data - y_pred) ** 2))
        
        # 计算参数标准误差
        param_errors = np.sqrt(np.diag(pcov))
        
        return {
            'peak_index': peak_index,
            'r_squared': r_squared,
            'rmse': rmse,
            'params': popt,
            'param_errors': param_errors,
            'data_points': len(x_data)
        }
        
    except Exception:
        return None

def send_file_to_wechat(file_path, webhook_url):
    """发送文件到企业微信"""
    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
            base64_data = base64.b64encode(file_data).decode('utf-8')
            md5_data = hashlib.md5(file_data).hexdigest()
        
        payload = {
            "msgtype": "file",
            "file": {
                "base64": base64_data,
                "md5": md5_data
            }
        }
        
        response = requests.post(webhook_url, json=payload)
        if response.status_code != 200:
            print(f"发送文件失败: {response.text}")
    except Exception as e:
        print(f"发送文件时出错: {e}")

def send_image_to_wechat(image_path, webhook_url):
    """发送图片到企业微信"""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            base64_data = base64.b64encode(image_data).decode('utf-8')
            md5_data = hashlib.md5(image_data).hexdigest()
        
        payload = {
            "msgtype": "image",
            "image": {
                "base64": base64_data,
                "md5": md5_data
            }
        }
        
        response = requests.post(webhook_url, json=payload)
        if response.status_code != 200:
            print(f"发送失败: {response.text}")
    except Exception as e:
        print(f"发送图片时出错: {e}")

def send_text_to_wechat(text, webhook_url):
    """发送文本消息到企业微信"""
    try:
        payload = {
            "msgtype": "text",
            "text": {
                "content": text
            }
        }
        
        response = requests.post(webhook_url, json=payload)
        if response.status_code != 200:
            print(f"发送文本失败: {response.text}")
    except Exception as e:
        print(f"发送文本时出错: {e}")

def analyze_focus_main(base_path, name, adjust=0):
    """主函数：分析聚焦度并推送图片"""
    # 构建processed文件夹路径
    processed_path = os.path.join(base_path, "processed")
    
    if not os.path.exists(processed_path):
        print(f"错误：找不到processed文件夹：{processed_path}")
        return None
    
    # 分析top和bottom文件夹
    top_folder = os.path.join(processed_path, "top")
    bottom_folder = os.path.join(processed_path, "bottom")
    
    top_result = analyze_folder_focus(top_folder)
    bottom_result = analyze_folder_focus(bottom_folder)
    
    # 提取峰值
    top_peak = top_result['peak_index'] if top_result else None
    bottom_peak = bottom_result['peak_index'] if bottom_result else None
    
    # 计算孔深
    aperture = None
    if top_peak is not None and bottom_peak is not None:
        aperture = (bottom_peak - top_peak) * 0.005 - adjust
    
    # 输出结果
    if top_peak is not None:
        print(f"TOP最佳: {top_peak}")
    else:
        print("TOP最佳: 无法计算")
    
    if bottom_peak is not None:
        print(f"BOTTOM最佳: {bottom_peak}")
    else:
        print("BOTTOM最佳: 无法计算")
    
    if aperture is not None:
        print(f"孔深: {aperture:.5f}")
    else:
        print("孔深: 无法计算")
    
    # 企业微信推送
    webhook_url = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=0c14a35f-f9df-42e3-8f3c-a76f28f1fbe5"
    
    # 获取当前时分
    current_time = datetime.now().strftime("%H:%M")
    
    # 先发送数据处理结果
    result_text = f"样品 {name}_{current_time}\n\n聚焦度分析结果：\n"
    if top_peak is not None:
        result_text += f"TOP最佳: {top_peak}\n"
        if top_result:
            result_text += f"TOP拟合R²: {top_result['r_squared']:.4f}\n"
            result_text += f"TOP拟合RMSE: {top_result['rmse']:.4f}\n"
            result_text += f"TOP数据点数: {top_result['data_points']}\n"
    else:
        result_text += "TOP最佳: 无法计算\n"
    
    if bottom_peak is not None:
        result_text += f"BOTTOM最佳: {bottom_peak}\n"
        if bottom_result:
            result_text += f"BOTTOM拟合R²: {bottom_result['r_squared']:.4f}\n"
            result_text += f"BOTTOM拟合RMSE: {bottom_result['rmse']:.4f}\n"
            result_text += f"BOTTOM数据点数: {bottom_result['data_points']}\n"
    else:
        result_text += "BOTTOM最佳: 无法计算\n"
    
    if aperture is not None:
        result_text += f"孔深: {aperture:.5f}"
    else:
        result_text += "孔深: 无法计算"
    
    send_text_to_wechat(result_text, webhook_url)
    
    # 再发送图片
    if top_peak is not None:
        try:
            original_dir = os.path.join(base_path, 'original')
            center_frame = int(round(top_peak))
            indices = [center_frame + i for i in range(-2, 3)]
            
            for idx in indices:
                image_path = os.path.join(original_dir, f"{idx}.png")
                if os.path.exists(image_path):
                    send_image_to_wechat(image_path, webhook_url)
        except Exception as e:
            print(f"推送图片时出错: {e}")
    
    return {
        'top_peak': top_peak,
        'bottom_peak': bottom_peak,
        'aperture': aperture,
        'adjust': adjust,
        'top_result': top_result,
        'bottom_result': bottom_result
    }

# 主要调用接口
def analyze_focus(base_path, name, adjust=0):
    """简化的调用接口"""
    return analyze_focus_main(base_path, name, adjust=adjust)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='聚焦度分析工具')
    parser.add_argument('--path', '-p', type=str, required=True, 
                       help='基础路径，程序会在此路径下查找processed文件夹')
    parser.add_argument('--name', '-n', type=str, required=True, 
                       help='样品名称，用于ZIP文件命名和微信消息标识')
    parser.add_argument('--adjust', '-a', type=float, default=0, 
                       help='孔深计算调整值，默认为0')
    
    args = parser.parse_args()
    
    result = analyze_focus_main(base_path=args.path, name=args.name, adjust=args.adjust)
    print(f"\n分析完成！结果: {result}")