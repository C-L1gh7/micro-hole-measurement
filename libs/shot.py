import os
import glob
import cv2
import numpy as np

def draw_crosshair(frame, crop_size=50, patch_size=50):
    """在原图上绘制中心十字线、crop框和patch区域"""
    # 拷贝原始帧，防止修改原图
    frame_with_overlay = frame.copy()
    h, w = frame_with_overlay.shape[:2]

    # 中心坐标
    center_x = w // 2
    center_y = h // 2

    # crop 区域框
    half_crop = crop_size // 2
    crop_x1 = center_x - half_crop
    crop_y1 = center_y - half_crop
    crop_x2 = center_x + half_crop
    crop_y2 = center_y + half_crop

    # patch 大小
    half_patch = patch_size // 2

    # 中心 patch
    cx1 = center_x - half_patch+30
    cy1 = center_y - half_patch
    cx2 = center_x + half_patch+30
    cy2 = center_y + half_patch

    # 左上角 patch
    top_x1 = crop_x1
    top_y1 = crop_y1
    top_x2 = crop_x1 + patch_size
    top_y2 = crop_y1 + patch_size

    # 十字线
    cv2.line(frame_with_overlay, (0, center_y), (w, center_y), (0, 0, 255), 1)
    cv2.line(frame_with_overlay, (center_x, 0), (center_x, h), (0, 0, 255), 1)

    # 画框
    cv2.rectangle(frame_with_overlay, (crop_x1, crop_y1), (crop_x2, crop_y2), (255, 0, 255), 2)  # crop 框
    cv2.rectangle(frame_with_overlay, (cx1, cy1), (cx2, cy2), (255, 0, 255), 2)                  # 中心 patch
    cv2.rectangle(frame_with_overlay, (top_x1, top_y1), (top_x2, top_y2), (255, 0, 255), 2)      # 左上角 patch

    return frame_with_overlay


def process_images(input_dir="photo", output_dir="processed", crop_size=50, patch_size=10):
    """灰度化原图 -> 保存至processed/gray -> 提取中心与左上角patch并保存"""
    img_paths = glob.glob(f"{input_dir}/**/original/*.png", recursive=True)
    if not img_paths:
        print(f"未找到 {input_dir} 目录下的图片！")
        return

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue

        # 灰度化
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 保存灰度图到 processed/gray
        # base_name = os.path.splitext(os.path.basename(img_path))[0]
        # base_processed_dir = os.path.dirname(img_path).replace("original", "processed")
        # gray_dir = os.path.join(base_processed_dir, "gray")
        # os.makedirs(gray_dir, exist_ok=True)
        # gray_path = os.path.join(gray_dir, f"{base_name}.png")
        # cv2.imwrite(gray_path, gray_img)

        # 接下来对灰度图进行裁切和patch提取
        h, w = gray_img.shape
        center_x, center_y = w // 2, h // 2
        half_crop = crop_size // 2

        crop_x1 = max(0, center_x - half_crop)
        crop_y1 = max(0, center_y - half_crop)
        crop_x2 = min(w, center_x + half_crop)
        crop_y2 = min(h, center_y + half_crop)

        crop_center_x = (crop_x1 + crop_x2) // 2
        crop_center_y = (crop_y1 + crop_y2) // 2
        half_patch = patch_size // 2

        # 中心patch
        center_patch = gray_img[
            crop_center_y - half_patch : crop_center_y + half_patch,
            crop_center_x - half_patch + 30 : crop_center_x + half_patch + 30
        ]

        # 左上角patch
        top_patch = gray_img[
            crop_y1 : crop_y1 + patch_size,
            crop_x1 : crop_x1 + patch_size
        ]

        # 保存到 bottom（中心 patch）
        bottom_dir = os.path.join(base_processed_dir, "bottom")
        os.makedirs(bottom_dir, exist_ok=True)
        cv2.imwrite(os.path.join(bottom_dir, f"{base_name}.png"), center_patch)

        # 保存到 top（左上角 patch）
        top_dir = os.path.join(base_processed_dir, "top")
        os.makedirs(top_dir, exist_ok=True)
        cv2.imwrite(os.path.join(top_dir, f"{base_name}.png"), top_patch)

    print("所有图片处理完成！")
    print("- 灰度图保存至 processed/gray")
    print("- 中心像素保存至 processed/bottom")
    print("- 左上角像素保存至 processed/top")



# 示例用法
if __name__ == "__main__":
    process_images(input_dir="photo", crop_size=500, patch_size=10)