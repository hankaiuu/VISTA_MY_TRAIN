from dataset import VTKG
from model import VISTA
from tqdm import tqdm
from utils import calculate_rank, metrics
import numpy as np
import argparse
import torch
import torch.nn as nn
import datetime
import time
import os
import copy
import math
import random
import distutils
import logging
from transformers import CLIPProcessor, CLIPModel
import requests
from PIL import Image, UnidentifiedImageError

OMP_NUM_THREADS = 8
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
torch.cuda.empty_cache()

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_format)
logger.addHandler(stream_handler)


parser = argparse.ArgumentParser()
parser.add_argument("--data", default="vista_data", type=str)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--dim", default=256, type=int)
parser.add_argument("--test_epoch", default=150, type=int)
parser.add_argument("--valid_epoch", default=50, type=int)
parser.add_argument("--exp", default="best")
parser.add_argument("--no_write", action="store_true")
parser.add_argument("--num_layer_enc_ent", default=2, type=int)
parser.add_argument("--num_layer_enc_rel", default=1, type=int)
parser.add_argument("--num_layer_dec", default=2, type=int)
parser.add_argument("--num_head", default=4, type=int)
parser.add_argument("--hidden_dim", default=2048, type=int)
parser.add_argument("--dropout", default=0.01, type=float)
parser.add_argument("--emb_dropout", default=0.9, type=float)
parser.add_argument("--vis_dropout", default=0.4, type=float)
parser.add_argument("--txt_dropout", default=0.1, type=float)
parser.add_argument("--smoothing", default=0.0, type=float)
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--decay", default=0.0, type=float)
parser.add_argument("--max_img_num", default=3, type=int)
parser.add_argument("--step_size", default=50, type=int)
args = parser.parse_args()

file_format = ""

for arg_name in vars(args).keys():
    if arg_name not in ["data", "exp", "no_write", "test_epoch"]:
        file_format += f"_{vars(args)[arg_name]}"

os.makedirs(f"./logs_test/{args.exp}/{args.data}", exist_ok=True)

file_handler = logging.FileHandler(
    f"./logs_test/{args.exp}/{args.data}/{file_format}.log"
)
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)

logger.info(f"{os.getpid()}")

KG = VTKG(args.data, logger, max_vis_len=args.max_img_num)

KG_Loader = torch.utils.data.DataLoader(KG, batch_size=args.batch_size, shuffle=True)

# 获取设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 修改模型初始化
model = VISTA(
    num_ent=KG.num_ent,
    num_rel=KG.num_rel,
    ent_vis=KG.ent_vis_matrix if hasattr(KG, "ent_vis_matrix") else None,
    rel_vis=KG.rel_vis_matrix,
    dim_vis=KG.vis_feat_size,
    ent_txt=KG.ent_txt_matrix,
    rel_txt=KG.rel_txt_matrix if hasattr(KG, "rel_txt_matrix") else None,
    dim_txt=KG.txt_feat_size,
    ent_vis_mask=KG.ent_vis_mask if hasattr(KG, "ent_vis_mask") else None,
    rel_vis_mask=KG.rel_vis_mask,
    dim_str=args.dim,
    num_head=args.num_head,
    dim_hid=args.hidden_dim,
    num_layer_enc_ent=args.num_layer_enc_ent,
    num_layer_enc_rel=args.num_layer_enc_rel,
    num_layer_dec=args.num_layer_dec,
    dropout=args.dropout,
    emb_dropout=args.emb_dropout,
    vis_dropout=args.vis_dropout,
    txt_dropout=args.txt_dropout,
).to(device)

# 修改模型加载
loaded_ckpt = torch.load(
    f"ckpt/best/VTKG-I/_0.0001_256_50_4_1_2_4_768_0.01_0.7_0.4_0.1_0.0_128_0.0_5_50_150.ckpt",
    map_location=device,
    weights_only=True,
)
model.load_state_dict(loaded_ckpt["model_state_dict"])

# 修改张量创建
all_ents = torch.arange(KG.num_ent).to(device)
all_rels = torch.arange(KG.num_rel).to(device)

# 修改特征映射器
feature_mapper = FeatureMapper().to(device)

# 修改CLIP模型
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# 使用与训练时相同的处理流程
def process_new_image(image_path):
    # 加载图像
    image = Image.open(image_path).convert("RGB")

    # 使用相同的CLIP处理器
    inputs = clip_processor(images=image, return_tensors="pt").to("cuda")

    # 使用相同的CLIP模型和特征映射层
    with torch.no_grad():
        # 获取CLIP原始特征
        clip_feature = clip_model.get_image_features(**inputs)  # 形状 [1, 512]

        # 通过特征映射层
        _, mapped_feature = feature_mapper(
            image_features=clip_feature
        )  # 取元组的第二个元素

        # 调整维度
        mapped_feature = mapped_feature.squeeze(0)  # 从 [1, 2304] 变为 [2304]

    # 检查特征是否有效
    assert not torch.isnan(mapped_feature).any(), "特征包含 NaN"
    assert not torch.isinf(mapped_feature).any(), "特征包含 Inf"

    return mapped_feature.cpu()


# test_feature = process_new_image("/media/sda3/jqz-workspace/Car_design_srtp/car_image/image.png")
# print(test_feature.shape)  # 应该输出 torch.Size([2304])

# 进行预测
model.eval()

# Define the input image folder and output folder
input_image_folder = "/Users/hankailu/Desktop/VISTA_my_train/car_img"
output_result_folder = (
    "/Users/hankailu/Desktop/VISTA_my_train/img_result/small_data_result_1000epoch"
)

# Create output folder if it doesn't exist
os.makedirs(output_result_folder, exist_ok=True)

# Get all image files from the input folder
image_files = [
    f
    for f in os.listdir(input_image_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
]

tail_entity = "整体外观"

# Process each image and save results
for img_file in image_files:
    img_path = os.path.join(input_image_folder, img_file)
    try:
        # Process the image
        new_image_feature = process_new_image(img_path)

        # Add new relation to KG (we'll use the image filename as relation name)
        rel_name = os.path.splitext(img_file)[0]  # Remove file extension
        KG.add_new_relation(
            rel_name=rel_name,
            image_features=new_image_feature.unsqueeze(0),  # Add batch dimension
        )

        # Get new relation features
        new_rel_feats = KG.rel_vis_matrix[-1]  # Get the last added feature

        # Create output file path using image name
        output_file_path = os.path.join(
            output_result_folder, f"predict_result_{rel_name}_{tail_entity}.txt"
        )

        # Make predictions and save results
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            with torch.no_grad():
                scores = (
                    model.predict_new_relation(
                        tail_id=KG.ent2id[tail_entity],
                        new_rel_features=new_rel_feats.unsqueeze(0),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                # Get all entity IDs
                all_ent_ids = torch.arange(KG.num_ent).cpu().numpy()

                # Pair scores with entity IDs
                score_ent_pairs = list(zip(scores[0], all_ent_ids))

                # Sort by score descending
                sorted_pairs = sorted(score_ent_pairs, key=lambda x: x[0], reverse=True)

                # Get top 10 results
                top_10_pairs = sorted_pairs[:10]

                # Write results to file
                output_file.write(f"Image: {img_file}\n")
                output_file.write("Top 10 predicted emotions:\n")
                for rank, (score, ent_id) in enumerate(top_10_pairs, start=1):
                    head_entity = KG.id2ent[ent_id]
                    output_file.write(f"{rank}. {head_entity} (score: {score:.4f})\n")

                # # Also save scores as numpy array
                # np.savetxt(os.path.join(output_result_folder, f"scores_{rel_name}.txt"),
                #           scores, fmt='%.4f')

        print(f"Processed {img_file} - results saved to {output_file_path}")

        # Remove the temporary relation we added (so we can add the next one)
        KG.remove_last_relation()

    except Exception as e:
        print(f"Error processing {img_file}: {str(e)}")

print("All images processed.")
