
#利用generate_captions.py加载模型生成图像caption

python /media/ubuntu/data/xican/hall_eval/CHAIR/generate_captions.py \
  --model_dir /path/to/model \
  --image_dir /path/to/val2014 \
  --output_json /path/to/hall_eval/CHAIR/result/coco_2017/generated_captions.json \
  --num_samples 500 \
  --device cuda \
  --prompt "Please describe this image in detail."

#使用chair.py生成评测结果

python chair.py \
    --cap_file /path/to/hall_eval/CHAIR/result/coco_2017/generated_captions.json \
    --annotation_path /path/to/hall_eval/CHAIR/annotations \
    --synonyms_file /path/to/hall_eval/CHAIR/synonyms.txt

或使用统一封装脚本（推荐）：后两项参数表示是否使用vcd或inter模块，不添加便是不使用；模型根据本地模型路径填写;图片在/home/tos_data/LLM_HM_3_model/halleval/CHAIR/val2014_1000中。对应的annotations是/home/tos_data/LLM_HM_3_model/halleval/CHAIR/annotations_1000。其余文件项目中都有包含，修改地址即可。

python /media/ubuntu/data/xican/hall_eval/CHAIR/run_eval.py \ 
  --model_dir /path/to/model \
  --image_dir /path/to/val2014 \
  --annotation_path /path/to/hall_eval/CHAIR/annotations \
  --synonyms_file /path/to/hall_eval/CHAIR/synonyms.txt \
  --result_root /path/to/hall_eval/CHAIR/result \
  --num_samples 500 \
  --device cuda \
  --prompt "Please describe this image in detail." \
  --use_vcd \
  --use_inter

生成的文件位置说明：
- captions: `/path/to/hall_eval/CHAIR/result/<模型名>/generated_captions.json`
- 评测结果: `/path/to/hall_eval/CHAIR/result/<模型名>/chair_results.json`