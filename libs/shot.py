import os
import glob
import cv2
import numpy as np

def draw_crosshair(frame, crop_size=50, patch_size=10):
    """绘制中心十字线和紫色区域框（crop_size区域内的中心+左上角）"""
    frame_with_overlay = cv2.resize(frame.copy(), (1280, 720))
    h, w = frame_with_overlay.shape[:2]
    center_x, center_y = w // 2, h // 2

    # 绘制红色十字线
    cv2.line(frame_with_overlay, (0, center_y), (w, center_y), (0, 0, 255), 1)
    cv2.line(frame_with_overlay, (center_x, 0), (center_x, h), (0, 0, 255), 1)

    # 绘制crop_size区域的紫色框
    half_crop = crop_size // 2
    crop_x1 = center_x - half_crop
    crop_y1 = center_y - half_crop
    crop_x2 = center_x + half_crop
    crop_y2 = center_y + half_crop
    cv2.rectangle(frame_with_overlay, (crop_x1, crop_y1), (crop_x2, crop_y2), (255, 0, 255), 2)

    # 在crop_size区域内绘制中心10x10的紫色框
    half_patch = patch_size // 2
    center_patch_x1 = center_x - half_patch
    center_patch_y1 = center_y - half_patch
    center_patch_x2 = center_x + half_patch
    center_patch_y2 = center_y + half_patch
    cv2.rectangle(frame_with_overlay, (center_patch_x1, center_patch_y1), (center_patch_x2, center_patch_y2), (255, 0, 255), 2)

    # 在crop_size区域内绘制左上角10x10的紫色框
    left_top_x1 = crop_x1
    left_top_y1 = crop_y1
    left_top_x2 = crop_x1 + patch_size
    left_top_y2 = crop_y1 + patch_size
    cv2.rectangle(frame_with_overlay, (left_top_x1, left_top_y1), (left_top_x2, left_top_y2), (255, 0, 255), 2)

    return frame_with_overlay


def process_images(input_dir="photo", output_dir="processed", crop_size=50, patch_size=10):
    """一步完成：裁切到crop_size，然后提取中心和左上角的10x10区域"""
    img_paths = glob.glob(f"{input_dir}/**/original/*.png", recursive=True)
    if not img_paths:
        print(f"未找到 {input_dir} 目录下的图片！")
        return

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue

        h, w = img.shape[:2]
        if h < crop_size or w < crop_size:
            print(f"跳过尺寸过小的图片: {img_path}")
            continue

        # 计算原图中心位置
        center_x, center_y = w // 2, h // 2
        half_crop = crop_size // 2
        
        # 计算crop_size区域的边界
        crop_x1 = center_x - half_crop
        crop_y1 = center_y - half_crop
        crop_x2 = center_x + half_crop
        crop_y2 = center_y + half_crop
        
        # 确保边界不超出图片范围
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(w, crop_x2)
        crop_y2 = min(h, crop_y2)

        # 一步完成：直接从原图计算并提取crop_size区域内的中心10x10像素
        # 中心10x10在crop_size区域的中心位置
        crop_center_x = (crop_x1 + crop_x2) // 2
        crop_center_y = (crop_y1 + crop_y2) // 2
        half_patch = patch_size // 2
        
        center_patch_x1 = crop_center_x - half_patch
        center_patch_y1 = crop_center_y - half_patch
        center_patch_x2 = crop_center_x + half_patch
        center_patch_y2 = crop_center_y + half_patch
        
        center_patch = img[center_patch_y1:center_patch_y2, center_patch_x1:center_patch_x2]

        # 一步完成：直接从原图计算并提取crop_size区域内的左上角10x10像素
        left_top_x1 = crop_x1
        left_top_y1 = crop_y1
        left_top_x2 = crop_x1 + patch_size
        left_top_y2 = crop_y1 + patch_size
        
        top_patch = img[left_top_y1:left_top_y2, left_top_x1:left_top_x2]

        # 检查提取的patch是否为空或尺寸不正确
        if center_patch.size == 0 or center_patch.shape[:2] != (patch_size, patch_size):
            print(f"中心patch提取失败: {img_path}")
            continue
        
        if top_patch.size == 0 or top_patch.shape[:2] != (patch_size, patch_size):
            print(f"左上角patch提取失败: {img_path}")
            continue

        # 灰度化
        center_patch_gray = cv2.cvtColor(center_patch, cv2.COLOR_BGR2GRAY)
        top_patch_gray = cv2.cvtColor(top_patch, cv2.COLOR_BGR2GRAY)

        # 创建输出文件夹
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 获取基础路径，将original替换为processed
        base_processed_dir = os.path.dirname(img_path).replace("original", "processed")
        
        # 保存到"processed/bottom"文件夹
        bottom_dir = os.path.join(base_processed_dir, "bottom")
        os.makedirs(bottom_dir, exist_ok=True)
        cv2.imwrite(os.path.join(bottom_dir, f"{base_name}.png"), center_patch_gray)
        
        # 保存到"processed/top"文件夹
        top_dir = os.path.join(base_processed_dir, "top")
        os.makedirs(top_dir, exist_ok=True)
        cv2.imwrite(os.path.join(top_dir, f"{base_name}.png"), top_patch_gray)



    print(f"所有图片处理完成！")
    print(f"- 中心10x10像素保存至 processed/bottom 目录")
    print(f"- 左上角10x10像素保存至 processed/top 目录")


# 示例用法
if __name__ == "__main__":
    process_images(input_dir="photo", crop_size=500, patch_size=10)