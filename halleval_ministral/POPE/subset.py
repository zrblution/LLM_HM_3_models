import json
import os
import shutil

# 输入 JSON 文件路径
json_path = '/media/ubuntu/data/xican/hall_eval/POPE/gqa/output/gqa_pope_popular.json'  # 替换为你的 output.json 路径
# 源图片文件夹
src_folder = '/media/ubuntu/data/xican/images'
# 目标文件夹
dst_folder = '/media/ubuntu/data/xican/hall_eval/POPE/gqa/popular'  # 替换为你希望的输出文件夹

# 如果目标文件夹不存在，创建它
os.makedirs(dst_folder, exist_ok=True)

# 读取 output.json
with open(json_path, 'r') as f:
    data = json.load(f)

copied_count = 0  # 计数实际拷贝的图片
skipped_count = 0  # 计数已存在的图片

# 遍历每条记录，拷贝对应图片
for item in data:
    img_name = item['img']
    src_path = os.path.join(src_folder, img_name)
    dst_path = os.path.join(dst_folder, img_name)
    
    if os.path.exists(src_path):
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)  # copy2 保留原文件的元数据
            copied_count += 1
        else:
            skipped_count += 1  # 已存在，跳过
    else:
        print(f"警告: 源文件不存在 {src_path}")

print(f"已将 {copied_count} 张图片拷贝到 {dst_folder}")
if skipped_count > 0:
    print(f"跳过 {skipped_count} 张已存在的图片")
