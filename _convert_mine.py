"""临时脚本：把自训 full_sft 权重转为 transformers 格式，供 web_demo 使用。"""
import os
import sys

root = os.path.dirname(os.path.abspath(__file__))  # minimind 根目录
sys.path.insert(0, root)
os.chdir(os.path.join(root, "scripts"))  # 让 ../out、../model 相对路径正确

import scripts.convert_model as cm
from model.model_minimind import MiniMindConfig

cm.lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=False)
cm.convert_torch2transformers_minimind("../out/full_sft_768.pth", "./minimind-mine")
print("转换完成，模型已输出到 scripts/minimind-mine/")
