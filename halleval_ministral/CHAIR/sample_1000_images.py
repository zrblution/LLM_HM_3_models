#!/usr/bin/env python3
"""
随机抽取1000张val2014图片及其对应的annotations
"""
import os
import json
import random
import shutil
from pathlib import Path

# 设置随机种子以保证可重复性
random.seed(42)

# 路径配置
SOURCE_IMAGE_DIR = "/home/tos_data/val2014"
TARGET_IMAGE_DIR = "/home/tos_data/LLM_HM_3_model/halleval/CHAIR/val2014_1000"
SOURCE_ANNO_DIR = "/home/tos_data/LLM_HM_3_model/halleval/CHAIR/annotations"
TARGET_ANNO_DIR = "/home/tos_data/LLM_HM_3_model/halleval/CHAIR/annotations_1000"

# 创建目标目录
os.makedirs(TARGET_IMAGE_DIR, exist_ok=True)
os.makedirs(TARGET_ANNO_DIR, exist_ok=True)

print("开始处理...")

# 1. 获取所有图片文件
all_images = [f for f in os.listdir(SOURCE_IMAGE_DIR) if f.endswith('.jpg')]
print(f"找到 {len(all_images)} 张图片")

# 2. 随机选择1000张图片
selected_images = random.sample(all_images, 1000)
print(f"随机选择了 {len(selected_images)} 张图片")

# 3. 复制图片
print("开始复制图片...")
for i, img_name in enumerate(selected_images, 1):
    src = os.path.join(SOURCE_IMAGE_DIR, img_name)
    dst = os.path.join(TARGET_IMAGE_DIR, img_name)
    shutil.copy2(src, dst)
    if i % 100 == 0:
        print(f"已复制 {i}/1000 张图片")

print("图片复制完成!")

# 4. 提取图片ID
selected_image_ids = set()
for img_name in selected_images:
    # 从文件名中提取ID，例如 COCO_val2014_000000123456.jpg -> 123456
    img_id = int(img_name.split('_')[-1].replace('.jpg', ''))
    selected_image_ids.add(img_id)

print(f"提取了 {len(selected_image_ids)} 个图片ID")

# 5. 处理annotations文件
annotation_files = [
    'captions_val2014.json',
    'instances_val2014.json',
    'person_keypoints_val2014.json'
]

for anno_file in annotation_files:
    print(f"\n处理 {anno_file}...")
    src_path = os.path.join(SOURCE_ANNO_DIR, anno_file)
    dst_path = os.path.join(TARGET_ANNO_DIR, anno_file)
    
    # 读取原始annotation
    with open(src_path, 'r') as f:
        data = json.load(f)
    
    # 过滤images
    filtered_images = [img for img in data['images'] if img['id'] in selected_image_ids]
    print(f"  过滤后的images数量: {len(filtered_images)}")
    
    # 过滤annotations
    filtered_annotations = [anno for anno in data['annotations'] if anno['image_id'] in selected_image_ids]
    print(f"  过滤后的annotations数量: {len(filtered_annotations)}")
    
    # 构建新的数据结构
    new_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'images': filtered_images,
        'annotations': filtered_annotations
    }
    
    # 如果有categories字段，也保留
    if 'categories' in data:
        new_data['categories'] = data['categories']
    
    # 保存新的annotation文件
    with open(dst_path, 'w') as f:
        json.dump(new_data, f)
    
    print(f"  已保存到 {dst_path}")

print("\n全部完成!")
print(f"图片保存在: {TARGET_IMAGE_DIR}")
print(f"Annotations保存在: {TARGET_ANNO_DIR}")
