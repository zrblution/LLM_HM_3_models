coco数据集-Qwen模型：
CUDA_VISIBLE_DEVICES= python /home/tos_data/LLM_HM_3_model/halleval/POPE/run_eval/run_eval_Qwen3_VL.py --dataset coco --model_dir /path/to/model(测评模型路径) （可选参数，用于降低幻觉）--use_vcd --use_inter

coco数据集-Ministral模型：
CUDA_VISIBLE_DEVICES= python /home/tos_data/LLM_HM_3_model/halleval/POPE/run_eval/run_eval_Ministral_VL.py --dataset coco --model_dir /path/to/model （可选参数，用于降低幻觉）--use_vcd --use_inter

gqa数据集-Qwen模型：
CUDA_VISIBLE_DEVICES= python /home/tos_data/LLM_HM_3_model/halleval/POPE/run_eval/run_eval_Qwen3_VL.py --dataset gqa --model_dir /path/to/model （可选参数，用于降低幻觉）--use_vcd --use_inter

gqa数据集-Ministral模型：
CUDA_VISIBLE_DEVICES= python /home/tos_data/LLM_HM_3_model/halleval/POPE/run_eval/run_eval_Ministral_VL.py --dataset gqa --model_dir /path/to/model （可选参数，用于降低幻觉）--use_vcd --use_inter