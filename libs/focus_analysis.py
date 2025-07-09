"""
聚焦度分析库
用于分析图像的聚焦度并找出最佳聚焦图片
"""

import cv2
import os
import numpy as np
from datetime import datetime
import glob
import re
import requests
import hashlib
import base64
import pywt


class FocusAnalyzer:
    """聚焦度分析器类"""
    
    def __init__(self, method='laplacian'):
        """
        初始化聚焦度分析器
        
        Args:
            method: 聚焦度计算方法 ('laplacian', 'sobel', 'brenner', 'variance', 'wavelet')
        """
        self.method = method
        self.supported_methods = ['laplacian', 'sobel', 'brenner', 'variance', 'wavelet']
        
        if method not in self.supported_methods:
            raise ValueError(f"不支持的方法: {method}. 支持的方法: {self.supported_methods}")
    
    def calculate_focus_measure(self, image):
        """
        计算图像的聚焦度
        
        Args:
            image: 输入图像
        
        Returns:
            focus_measure: 聚焦度值
        """
        # 转换为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if self.method == 'laplacian':
            # 拉普拉斯算子方法
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            focus_measure = np.var(laplacian)
        
        elif self.method == 'sobel':
            # Sobel算子方法
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            focus_measure = np.mean(sobel_x**2 + sobel_y**2)

        
        elif self.method == 'brenner':
            # Brenner梯度方法
            brenner = np.zeros_like(gray, dtype=np.float64)
            brenner[:-2, :] = np.abs(gray[2:, :].astype(np.float64) - gray[:-2, :].astype(np.float64))
            focus_measure = np.mean(brenner)
        
        elif self.method == 'variance':
            # 方差方法
            focus_measure = np.var(gray)
        
        elif self.method == 'wavelet':
        # 多层小波分解（默认3层，使用 db2）
            level = 3
            wavelet_name = 'db2'
            coeffs = pywt.wavedec2(gray, wavelet=wavelet_name, level=level)

            # coeffs[0] 是 LL，其余是 [(LH, HL, HH), ...]
            focus_measure = 0.0
            for i in range(1, len(coeffs)):
                cH, cV, cD = coeffs[i]
                focus_measure += np.sum(cH**2) + np.sum(cV**2) + np.sum(cD**2)

            # 可选归一化
            focus_measure /= gray.size
        
        return focus_measure
    
    def analyze_folder(self, folder_path, verbose=True):
        """
        分析文件夹中所有图片的聚焦度
        
        Args:
            folder_path: 文件夹路径
            verbose: 是否显示详细信息
        
        Returns:
            dict: 包含results(所有结果)和best(最佳结果)的字典
        """
        if not os.path.exists(folder_path):
            if verbose:
                print(f"文件夹不存在: {folder_path}")
            return {'results': [], 'best': None, 'error': f"文件夹不存在: {folder_path}"}
        
        # 支持的图像格式
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        if not image_files:
            if verbose:
                print(f"文件夹中没有找到图片: {folder_path}")
            return {'results': [], 'best': None, 'error': f"文件夹中没有找到图片: {folder_path}"}
        
        results = []
        best_focus = -1
        best_image = None
        
        if verbose:
            print(f"\n分析文件夹: {folder_path}")
            print(f"找到 {len(image_files)} 张图片")
            print("-" * 50)
        
        for image_path in image_files:
            try:
                # 读取图像
                image = cv2.imread(image_path)
                if image is None:
                    if verbose:
                        print(f"无法读取图像: {image_path}")
                    continue
                
                # 计算聚焦度
                focus_measure = self.calculate_focus_measure(image)
                
                # 提取文件名
                filename = os.path.basename(image_path)
                
                # 存储结果
                result = {
                    'filename': filename,
                    'path': image_path,
                    'focus_measure': focus_measure
                }
                results.append(result)
                
                # 更新最佳聚焦度
                if focus_measure > best_focus:
                    best_focus = focus_measure
                    best_image = result
                
                if verbose:
                    print(f"{filename}: {focus_measure:.2f}")
                
            except Exception as e:
                if verbose:
                    print(f"处理图像时出错 {image_path}: {e}")
        
        # 按聚焦度排序
        results.sort(key=lambda x: x['focus_measure'], reverse=True)
        
        return {'results': results, 'best': best_image, 'error': None}
    
    def analyze_top_bottom_folders(self, processed_base_path, verbose=True):
        """
        分析processed文件夹下的top和bottom子文件夹
        
        Args:
            processed_base_path: processed文件夹的基础路径
            verbose: 是否显示详细信息
        
        Returns:
            dict: 包含top和bottom分析结果的字典
        """
        top_folder = os.path.join(processed_base_path, "top")
        bottom_folder = os.path.join(processed_base_path, "bottom")
        
        if verbose:
            print(f"使用聚焦度计算方法: {self.method}")
        
        # 分析top文件夹
        top_analysis = self.analyze_folder(top_folder, verbose)
        
        # 分析bottom文件夹
        bottom_analysis = self.analyze_folder(bottom_folder, verbose)
        
        return {
            'top': top_analysis,
            'bottom': bottom_analysis,
            'method': self.method,
            'processed_path': processed_base_path
        }


def extract_number_from_filename(filename):
    """
    从文件名中提取数字编号
    
    Args:
        filename: 文件名
    
    Returns:
        int: 提取的数字，如果没有找到则返回None
    """
    # 使用正则表达式查找文件名中的数字
    numbers = re.findall(r'\d+', filename)
    if numbers:
        # 返回第一个找到的数字
        return int(numbers[0])
    return None


def calculate_aperture(top_best, bottom_best, adjust=0):
    """
    计算孔径值
    
    Args:
        top_best: top文件夹最佳聚焦图片信息
        bottom_best: bottom文件夹最佳聚焦图片信息
        adjust: 调整值，默认为0
    
    Returns:
        float: 孔径值，如果无法计算则返回None
    """
    if not top_best or not bottom_best:
        return None
    
    # 从文件名中提取编号
    top_number = extract_number_from_filename(top_best['filename'])
    bottom_number = extract_number_from_filename(bottom_best['filename'])
    
    if top_number is None or bottom_number is None:
        return None
    
    # 计算孔径：(bottom编号 - top编号) * 比例尺 - adjust
    aperture = (bottom_number - top_number) * 0.005 - adjust
    
    return aperture


def find_latest_photo_folder():
    """
    查找最新的photo文件夹
    
    Returns:
        str: 最新photo文件夹的路径，如果没有找到则返回None
    """
    photo_base = "photo"
    if not os.path.exists(photo_base):
        return None
    
    # 获取所有photo文件夹
    photo_folders = [f for f in os.listdir(photo_base) if os.path.isdir(os.path.join(photo_base, f))]
    
    if not photo_folders:
        return None
    
    # 按名称排序（由于格式是YYYYMMDD_HHMM，字符串排序就是时间排序）
    photo_folders.sort(reverse=True)
    
    return os.path.join(photo_base, photo_folders[0])


def get_best_focus_images(processed_path=None, method='laplacian', adjust=0, verbose=True):
    """
    获取最佳聚焦度的图片（简化的接口函数）
    
    Args:
        processed_path: processed文件夹路径，如果为None则自动查找
        method: 聚焦度计算方法
        adjust: 孔径计算调整值，默认为0
        verbose: 是否显示详细信息
    
    Returns:
        dict: 包含top和bottom最佳聚焦图片信息和孔径值的字典
    """
    # 确定processed文件夹路径
    if processed_path is None:
        latest_folder = find_latest_photo_folder()
        if latest_folder:
            processed_path = os.path.join(latest_folder, "processed")
        else:
            processed_path = "processed"
    
    # 创建分析器
    analyzer = FocusAnalyzer(method)
    
    # 执行分析
    analysis_result = analyzer.analyze_top_bottom_folders(processed_path, verbose)
    
    # 提取最佳图片信息
    top_best = analysis_result['top']['best']
    bottom_best = analysis_result['bottom']['best']
    
    # 计算孔径
    aperture = calculate_aperture(top_best, bottom_best, adjust)
    
    result = {
        'top_best': top_best,
        'bottom_best': bottom_best,
        'aperture': aperture,
        'method': method,
        'processed_path': processed_path,
        'adjust': adjust,
    }
    
    return result


def print_analysis_summary(analysis_result, save_to_file=True):
    """
    打印分析结果摘要
    
    Args:
        analysis_result: 分析结果字典
        save_to_file: 是否保存到文件
    """
    print("\n" + "=" * 60)
    print("分析结果汇总")
    print("=" * 60)
    
    top_best = analysis_result.get('top_best')
    bottom_best = analysis_result.get('bottom_best')
    aperture = analysis_result.get('aperture')
    adjust = analysis_result.get('adjust', 0)
    
    if top_best:
        print(f"\nTOP文件夹最佳聚焦图片:")
        print(f"文件名: {top_best['filename']}")
        print(f"聚焦度: {top_best['focus_measure']:.2f}")
        print(f"路径: {top_best['path']}")
        
        # 提取并显示编号
        top_number = extract_number_from_filename(top_best['filename'])
        if top_number is not None:
            print(f"编号: {top_number}")
    else:
        print("\nTOP文件夹: 未找到有效图片")
    
    if bottom_best:
        print(f"\nBOTTOM文件夹最佳聚焦图片:")
        print(f"文件名: {bottom_best['filename']}")
        print(f"聚焦度: {bottom_best['focus_measure']:.2f}")
        print(f"路径: {bottom_best['path']}")
        
        # 提取并显示编号
        bottom_number = extract_number_from_filename(bottom_best['filename'])
        if bottom_number is not None:
            print(f"编号: {bottom_number}")
    else:
        print("\nBOTTOM文件夹: 未找到有效图片")
    
    # 显示孔径计算结果
    if aperture is not None:
        print(f"\n孔径计算:")
        print(f"调整值 (adjust): {adjust}")
        print(f"孔径值: {aperture:.3f}")
    else:
        print(f"\n孔径计算: 无法计算（缺少有效的图片编号）")
    
    if save_to_file:
        save_analysis_to_file(analysis_result)
    
    print("=" * 60)


def save_analysis_to_file(analysis_result):
    """
    保存分析结果到文件
    
    Args:
        analysis_result: 分析结果字典
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"focus_analysis_{timestamp}.txt"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("图像聚焦度分析结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"计算方法: {analysis_result.get('method', 'unknown')}\n")
        f.write(f"分析文件夹: {analysis_result.get('processed_path', 'unknown')}\n")
        f.write(f"调整值 (adjust): {analysis_result.get('adjust', 0)}\n\n")
        
        top_best = analysis_result.get('top_best')
        bottom_best = analysis_result.get('bottom_best')
        aperture = analysis_result.get('aperture')
        
        if top_best:
            f.write("TOP文件夹最佳聚焦图片:\n")
            f.write(f"文件名: {top_best['filename']}\n")
            f.write(f"聚焦度: {top_best['focus_measure']:.2f}\n")
            f.write(f"路径: {top_best['path']}\n")
            
            top_number = extract_number_from_filename(top_best['filename'])
            if top_number is not None:
                f.write(f"编号: {top_number}\n")
            f.write("\n")
        
        if bottom_best:
            f.write("BOTTOM文件夹最佳聚焦图片:\n")
            f.write(f"文件名: {bottom_best['filename']}\n")
            f.write(f"聚焦度: {bottom_best['focus_measure']:.2f}\n")
            f.write(f"路径: {bottom_best['path']}\n")
            
            bottom_number = extract_number_from_filename(bottom_best['filename'])
            if bottom_number is not None:
                f.write(f"编号: {bottom_number}\n")
            f.write("\n")
        
        # 保存孔径计算结果
        if aperture is not None:
            f.write("孔径计算:\n")
            f.write(f"孔径值: {aperture:.5f}\n\n")
        else:
            f.write("孔径计算: 无法计算（缺少有效的图片编号）\n\n")
    
    print(f"结果已保存到: {result_file}")

def send_image_to_wechat(image_path, webhook_url):
    # 读取图片并编码
    with open(image_path, 'rb') as f:
        image_data = f.read()
        base64_data = base64.b64encode(image_data).decode('utf-8')
        md5_data = hashlib.md5(image_data).hexdigest()
    
    # 构造 payload
    payload = {
        "msgtype": "image",
        "image": {
            "base64": base64_data,
            "md5": md5_data
        }
    }
    
    # 发送 POST 请求
    response = requests.post(webhook_url, json=payload)
    if response.status_code != 200:
        print(f"发送失败: {response.text}")


# 简化的调用函数
def analyze_focus(processed_path=None, method='laplacian', adjust=0, verbose=True, save_file=True):
    """
    一键分析聚焦度（最简化的接口）
    
    Args:
        processed_path: processed文件夹路径
        method: 聚焦度计算方法
        adjust: 孔径计算调整值，默认为0
        verbose: 是否显示详细信息
        save_file: 是否保存结果到文件
    
    Returns:
        dict: 分析结果
    """
    result = get_best_focus_images(processed_path, method, adjust, verbose)
    
    if verbose:
        print_analysis_summary(result, save_file)
    
    # 企业微信推送5张原图：top最佳帧的前后两帧
    webhook_url = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=0c14a35f-f9df-42e3-8f3c-a76f28f1fbe5"
    top_best = result.get('top_best')

    if top_best:
        top_number = extract_number_from_filename(top_best['filename'])
        if top_number is not None:
            try:
                # 原图目录为 processed_path 的上一级的 original
                original_dir = os.path.join(os.path.dirname(result['processed_path']), 'original')

                # 构造前后两帧共5张的编号列表
                indices = [top_number + i for i in range(-2, 3)]

                for idx in indices:
                    image_path = os.path.join(original_dir, f"{idx}.png")
                    if os.path.exists(image_path):
                        send_image_to_wechat(image_path, webhook_url)
            except Exception as e:
                print(f"[警告] 尝试发送原图时出错: {e}")

    return result