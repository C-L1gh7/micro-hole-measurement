import cv2
import os
import numpy as np
from datetime import datetime
import glob
import re
import requests
import hashlib
import base64


class FocusAnalyzer:
    """聚焦度分析器类"""

    def __init__(self, method='laplacian'):
        """
        初始化聚焦度分析器
        Args:
            method: 聚焦度计算方法 ('laplacian', 'sobel', 'brenner', 'variance', 'tenengrad')
        """
        self.method = method
        self.supported_methods = ['laplacian', 'sobel', 'brenner', 'variance', 'tenengrad']

        if method not in self.supported_methods:
            raise ValueError(f"不支持的方法: {method}. 支持的方法: {self.supported_methods}")

    def calculate_focus_measure(self, image):
        """
        计算图像的聚焦度
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if self.method == 'laplacian':
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return np.var(laplacian)

        elif self.method == 'sobel':
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return np.mean(sobel_x**2 + sobel_y**2)

        elif self.method == 'brenner':
            brenner = np.zeros_like(gray, dtype=np.float64)
            brenner[:-2, :] = np.abs(gray[2:, :].astype(np.float64) - gray[:-2, :].astype(np.float64))
            return np.mean(brenner)

        elif self.method == 'variance':
            return np.var(gray)

        elif self.method == 'tenengrad':
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return np.mean(sobel_x**2 + sobel_y**2)

    def analyze_folder(self, folder_path, verbose=True):
        if not os.path.exists(folder_path):
            if verbose:
                print(f"文件夹不存在: {folder_path}")
            return {'results': [], 'best': None, 'error': f"文件夹不存在: {folder_path}"}

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
                image = cv2.imread(image_path)
                if image is None:
                    if verbose:
                        print(f"无法读取图像: {image_path}")
                    continue

                focus_measure = self.calculate_focus_measure(image)
                filename = os.path.basename(image_path)
                result = {'filename': filename, 'path': image_path, 'focus_measure': focus_measure}
                results.append(result)

                if focus_measure > best_focus:
                    best_focus = focus_measure
                    best_image = result

                if verbose:
                    print(f"{filename}: {focus_measure:.2f}")

            except Exception as e:
                if verbose:
                    print(f"处理图像时出错 {image_path}: {e}")

        results.sort(key=lambda x: x['focus_measure'], reverse=True)
        return {'results': results, 'best': best_image, 'error': None}

    def analyze_top_bottom_folders(self, processed_base_path, verbose=True):
        top_folder = os.path.join(processed_base_path, "top")
        bottom_folder = os.path.join(processed_base_path, "bottom")

        if verbose:
            print(f"使用聚焦度计算方法: {self.method}")

        top_analysis = self.analyze_folder(top_folder, verbose)
        bottom_analysis = self.analyze_folder(bottom_folder, verbose)

        return {
            'top': top_analysis,
            'bottom': bottom_analysis,
            'method': self.method,
            'processed_path': processed_base_path
        }


def extract_number_from_filename(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else None


def calculate_aperture(top_best, bottom_best, adjust=0):
    if not top_best or not bottom_best:
        return None
    top_number = extract_number_from_filename(top_best['filename'])
    bottom_number = extract_number_from_filename(bottom_best['filename'])
    if top_number is None or bottom_number is None:
        return None
    return (bottom_number - top_number) * 0.005 - adjust


def find_latest_photo_folder():
    photo_base = "photo"
    if not os.path.exists(photo_base):
        return None
    photo_folders = [f for f in os.listdir(photo_base) if os.path.isdir(os.path.join(photo_base, f))]
    if not photo_folders:
        return None
    photo_folders.sort(reverse=True)
    return os.path.join(photo_base, photo_folders[0])


def get_best_focus_images(processed_path=None, method='laplacian', adjust=0, verbose=True):
    if processed_path is None:
        latest_folder = find_latest_photo_folder()
        processed_path = os.path.join(latest_folder, "processed") if latest_folder else "processed"

    analyzer = FocusAnalyzer(method)
    analysis_result = analyzer.analyze_top_bottom_folders(processed_path, verbose)

    top_best = analysis_result['top']['best']
    bottom_best = analysis_result['bottom']['best']
    aperture = calculate_aperture(top_best, bottom_best, adjust)

    return {
        'top_best': top_best,
        'bottom_best': bottom_best,
        'aperture': aperture,
        'method': method,
        'processed_path': processed_path,
        'adjust': adjust
    }


def print_analysis_summary(analysis_result, save_to_file=True):
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
        bottom_number = extract_number_from_filename(bottom_best['filename'])
        if bottom_number is not None:
            print(f"编号: {bottom_number}")
    else:
        print("\nBOTTOM文件夹: 未找到有效图片")

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
                f.write(f"编号: {top_number}\n\n")

        if bottom_best:
            f.write("BOTTOM文件夹最佳聚焦图片:\n")
            f.write(f"文件名: {bottom_best['filename']}\n")
            f.write(f"聚焦度: {bottom_best['focus_measure']:.2f}\n")
            f.write(f"路径: {bottom_best['path']}\n")
            bottom_number = extract_number_from_filename(bottom_best['filename'])
            if bottom_number is not None:
                f.write(f"编号: {bottom_number}\n\n")

        if aperture is not None:
            f.write("孔径计算:\n")
            f.write(f"孔径值: {aperture:.3f}\n")
        else:
            f.write("孔径计算: 无法计算\n")

    print(f"结果已保存到: {result_file}")


def send_image_to_wechat(image_path, webhook_url):
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        base64_str = base64.b64encode(image_data).decode()
        md5_str = hashlib.md5(image_data).hexdigest()
        data = {
            "msgtype": "image",
            "image": {
                "base64": base64_str,
                "md5": md5_str
            }
        }
        response = requests.post(webhook_url, json=data)
        if response.status_code == 200:
            print(f"✅ 图片已成功发送到企业微信")
        else:
            print(f"❌ 发送失败: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"❌ 发送图片时出错: {e}")


def analyze_focus(processed_path=None, method='laplacian', adjust=0, verbose=True, save_file=True):
    result = get_best_focus_images(processed_path, method, adjust, verbose)
    if verbose:
        print_analysis_summary(result, save_file)

    # 替换为发送对应原图（original 文件夹）
    webhook_url = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=0c14a35f-f9df-42e3-8f3c-a76f28f1fbe5"
    if result['top_best']:
        original_path = os.path.join(
            result['processed_path'],  # processed 基路径
            "original",
            result['top_best']['filename']
        )
        if os.path.exists(original_path):
            send_image_to_wechat(original_path, webhook_url)
        else:
            print(f"⚠️ 未找到原图路径: {original_path}")

    return result

