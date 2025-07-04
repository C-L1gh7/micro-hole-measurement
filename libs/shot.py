import os
import glob
import cv2
import numpy as np

def draw_crosshair(frame, crop_size=50, patch_size=10):
    """绘制中心十字线和紫色区域框（中心+左上角）"""
    frame_with_overlay = cv2.resize(frame.copy(), (1280, 720))
    h, w = frame_with_overlay.shape[:2]
    center_x, center_y = w // 2, h // 2

    # 绘制红色十字线
    cv2.line(frame_with_overlay, (0, center_y), (w, center_y), (0, 0, 255), 1)
    cv2.line(frame_with_overlay, (center_x, 0), (center_x, h), (0, 0, 255), 1)

    # 原图中心区域（紫色框）
    half_patch = patch_size // 2
    top_left_center = (center_x - half_patch, center_y - half_patch)
    bottom_right_center = (center_x + half_patch, center_y + half_patch)
    cv2.rectangle(frame_with_overlay, top_left_center, bottom_right_center, (255, 0, 255), 2)

    # 原图左上角区域（紫色框）
    cv2.rectangle(frame_with_overlay, (0, 0), (patch_size, patch_size), (255, 0, 255), 2)

    return frame_with_overlay


def process_images(input_dir="photo", output_dir="processed", crop_size=50, patch_size=10):
    """从原图中提取中心和左上角的10x10区域，灰度化并保存"""
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

        # ---- 中心10×10 区域 ----
        center_x, center_y = w // 2, h // 2
        half_patch = patch_size // 2
        center_patch = img[center_y - half_patch:center_y + half_patch,
                           center_x - half_patch:center_x + half_patch]

        # ---- 左上角10×10 区域 ----
        top_patch = img[0:patch_size, 0:patch_size]

        # 灰度化
        center_patch_gray = cv2.cvtColor(center_patch, cv2.COLOR_BGR2GRAY)
        top_patch_gray = cv2.cvtColor(top_patch, cv2.COLOR_BGR2GRAY)

        # 保存路径
        base_dir = os.path.dirname(img_path).replace("original", output_dir)
        os.makedirs(base_dir, exist_ok=True)

        # 保存两张图
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(base_dir, f"{base_name}_bottom_surface.png"), center_patch_gray)
        cv2.imwrite(os.path.join(base_dir, f"{base_name}_top_surface.png"), top_patch_gray)

        # 绘制紫色框供可视化
        visual = draw_crosshair(img, crop_size=crop_size, patch_size=patch_size)
        cv2.imwrite(os.path.join(base_dir, f"{base_name}_debug_overlay.png"), visual)

    print(f"所有图片处理完成，保存至 {output_dir} 目录！")
