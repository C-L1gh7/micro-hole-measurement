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
    cx1 = center_x - half_patch
    cy1 = center_y - half_patch
    cx2 = center_x + half_patch
    cy2 = center_y + half_patch

    # 上方中心 patch（y轴不变，x轴移到正中心）
    top_x1 = center_x - half_patch
    top_y1 = crop_y1
    top_x2 = center_x + half_patch
    top_y2 = crop_y1 + patch_size

    # 十字线
    cv2.line(frame_with_overlay, (0, center_y), (w, center_y), (0, 0, 255), 1)
    cv2.line(frame_with_overlay, (center_x, 0), (center_x, h), (0, 0, 255), 1)

    # 画框
    cv2.rectangle(frame_with_overlay, (crop_x1, crop_y1), (crop_x2, crop_y2), (255, 0, 255), 2)  # crop 框
    cv2.rectangle(frame_with_overlay, (cx1, cy1), (cx2, cy2), (255, 0, 255), 2)                  # 中心 patch
    cv2.rectangle(frame_with_overlay, (top_x1, top_y1), (top_x2, top_y2), (255, 0, 255), 2)      # 上方中心 patch

    return frame_with_overlay


def process_images(base_path, crop_size=50, patch_size=10, k_pressed_photo_number=None):
    """
    处理指定路径下original文件夹中的图片
    灰度化原图 -> 保存至processed/gray -> 提取中心与上方中心patch并保存
    如果k_pressed_photo_number不为None，则根据照片编号分配到top/bottom文件夹
    """
    # 构建original文件夹路径
    original_dir = os.path.join(base_path, "original")
    
    if not os.path.exists(original_dir):
        print(f"错误：找不到original文件夹：{original_dir}")
        return
    
    # 查找original文件夹中的图片
    img_paths = glob.glob(os.path.join(original_dir, "*.png"))
    if not img_paths:
        print(f"未找到 {original_dir} 目录下的图片！")
        return

    # 创建processed文件夹
    processed_dir = os.path.join(base_path, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue

        # 获取图片的编号（从文件名获取）
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        try:
            photo_number = int(base_name)
        except ValueError:
            print(f"无法解析图片编号: {base_name}")
            continue

        # 接下来对原图进行裁切和patch提取
        h, w = img.shape[:2]
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
        center_patch = img[
            crop_center_y - half_patch : crop_center_y + half_patch,
            crop_center_x - half_patch : crop_center_x + half_patch
        ]

        # 上方中心patch（y轴保持在crop区域上边界，x轴移到正中心）
        top_patch = img[
            crop_y1 : crop_y1 + patch_size,
            center_x - half_patch : center_x + half_patch
        ]

        # 根据k_pressed_photo_number决定保存位置
        if k_pressed_photo_number is not None:
            # 编号小于k_pressed_photo_number的图片：上方中心patch保存到top，中心patch保存到bottom
            # 编号大于等于k_pressed_photo_number的图片：中心patch保存到bottom，上方中心patch保存到top
            if photo_number <= k_pressed_photo_number:
                # 上方中心patch -> top
                top_dir = os.path.join(processed_dir, "top")
                os.makedirs(top_dir, exist_ok=True)
                cv2.imwrite(os.path.join(top_dir, f"{base_name}.png"), top_patch)
            else:
                # 中心patch -> bottom
                bottom_dir = os.path.join(processed_dir, "bottom")
                os.makedirs(bottom_dir, exist_ok=True)
                cv2.imwrite(os.path.join(bottom_dir, f"{base_name}.png"), center_patch)
        else:
            # 如果没有按过k键，使用原来的逻辑
            # 保存到 bottom（中心 patch）
            bottom_dir = os.path.join(processed_dir, "bottom")
            os.makedirs(bottom_dir, exist_ok=True)
            cv2.imwrite(os.path.join(bottom_dir, f"{base_name}.png"), center_patch)

            # 保存到 top（上方中心 patch）
            top_dir = os.path.join(processed_dir, "top")
            os.makedirs(top_dir, exist_ok=True)
            cv2.imwrite(os.path.join(top_dir, f"{base_name}.png"), top_patch)

    print("所有图片处理完成！")
    print(f"处理路径: {base_path}")
    print(f"输出路径: {processed_dir}")
    
    if k_pressed_photo_number is not None:
        print(f"根据k键按下时的照片编号({k_pressed_photo_number})进行分类处理")
        print(f"- 编号 <= {k_pressed_photo_number} 的图片：上方中心patch保存至top")
        print(f"- 编号 > {k_pressed_photo_number} 的图片：中心patch保存至bottom")
    else:
        print("- 中心像素保存至 processed/bottom")
        print("- 上方中心像素保存至 processed/top")


# 示例用法
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='图像处理工具')
    parser.add_argument('--path', '-p', type=str, required=True, 
                       help='基础路径，程序会在此路径下的original文件夹中查找图片')
    parser.add_argument('--crop_size', '-c', type=int, default=500, 
                       help='裁切区域大小，默认为500')
    parser.add_argument('--patch_size', '-s', type=int, default=10, 
                       help='patch大小，默认为10')
    parser.add_argument('--k_pressed', '-k', type=int, default=None, 
                       help='k键按下时的照片编号，用于分类处理')
    
    args = parser.parse_args()
    
    process_images(
        base_path=args.path,
        crop_size=args.crop_size,
        patch_size=args.patch_size,
        k_pressed_photo_number=args.k_pressed
    )