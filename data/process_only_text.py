import json
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import random
import os
import shutil

# 获取文件夹中所有的 JSON 文件
input_dir = '/media/sda3/jqz-workspace/Car_design_srtp_data/data'  # 修改为存放 JSON 文件的文件夹路径
# 设置输出文件夹路径
output_dir = '/media/sda3/jqz-workspace/Car_design_srtp/VISTA/VISTA_my_train/data/vista_data_all'  # 修改为你想要保存文件的路径


# 清空目标文件夹
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# 创建文件夹（如果不存在的话）
os.makedirs(output_dir, exist_ok=True)

# 加载预训练的 CLIP 模型和处理器
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 定义全连接层以调整 CLIP 输出的维度
class FeatureMapper(nn.Module):
    def __init__(self):
        super(FeatureMapper, self).__init__()
        # 将 CLIP 文本特征 (512) 映射到 BERT 文本特征 (768)
        self.text_fc = nn.Linear(512, 768)
        # 将 CLIP 图像特征 (512) 映射到 ResNet 图像特征 (2304)
        self.image_fc = nn.Linear(512, 2304)

    def forward(self, text_features=None, image_features=None):
        if text_features is not None:
            text_features = self.text_fc(text_features)
        if image_features is not None:
            image_features = self.image_fc(image_features)
        return text_features, image_features

# 创建一个特征映射器实例
feature_mapper = FeatureMapper().to("cuda")

def get_clip_text_embedding(text):
    """使用 CLIP 提取文本特征，并调整维度"""
    inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():  # 禁用梯度计算
        outputs = clip_model.get_text_features(**inputs)  # 形状是 (1, 512)
    # 使用全连接层调整维度
    text_features, _ = feature_mapper(text_features=outputs)
    return text_features.squeeze().cpu().detach()  # 确保没有 requires_grad

def get_clip_image_embedding(image_url):
    """使用 CLIP 提取图像特征，并调整维度"""
    try:
        # 发送 HTTP 请求获取图像数据
        response = requests.get(image_url, timeout=10)
        
        # 验证响应状态码和内容类型
        if response.status_code != 200:
            raise ValueError(f"Invalid HTTP response: Status Code {response.status_code}")
        if 'image' not in response.headers.get('Content-Type', ''):
            raise ValueError(f"Invalid Content-Type: {response.headers.get('Content-Type')}")

        # 尝试加载图像
        try:
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except UnidentifiedImageError:
            raise ValueError("The response content is not a valid image.")

        # 使用 CLIP 处理图像
        inputs = clip_processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():  # 禁用梯度计算
            outputs = clip_model.get_image_features(**inputs)  # 形状是 (1, 512)
        # 使用全连接层调整维度
        _, image_features = feature_mapper(image_features=outputs)
        return image_features.cpu().detach()  # 确保没有 requires_grad

    except Exception as e:
        print(f"Error processing image from URL {image_url}: {e}")
        return None  # 如果图像无法加载，返回 None

# 替换原有的特征提取函数
get_text_embedding = get_clip_text_embedding
get_image_embedding = get_clip_image_embedding



# 初始化数据结构
entities = set()
relations = set()
triplets = []
valid_triplets = []
ent2txt = {}  # 用于存储实体的文本特征
rel2vis = {}  # 用于存储关系的视觉特征

# 遍历文件夹中的所有 JSON 文件
for filename in os.listdir(input_dir):
    print(filename)
    
    if filename.endswith(".json"):
        json_file_path = os.path.join(input_dir, filename)
        
        # 读取每个 JSON 文件
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 处理每一条记录
        for entry in data:
            sentiment_words = entry["情感词"]
            appearance_features = entry["外观特征"]
            images = entry["图片"]  # 假设每个条目只有一组图片

            for sentiment in sentiment_words:
                for appearance in appearance_features:
                    # 创建实体和关系
                    head_entity = sentiment
                    relation = appearance
                    tail_entity = appearance  # 假设关系是“情感词 -> 外观特征”

                    # 保存三元组
                    triplets.append((head_entity, relation, tail_entity))

                    # 保存实体和关系
                    entities.add(sentiment)
                    entities.add(appearance)
                    relations.add(appearance)

                    # 提取实体的文本特征
                    if sentiment not in ent2txt:
                        ent2txt[sentiment] = get_text_embedding(sentiment)
                    if appearance not in ent2txt:
                        ent2txt[appearance] = get_text_embedding(appearance)

                    # 提取关系的图像特征
                    for img in images:
                        if relation not in rel2vis:
                            embedding = get_image_embedding(img)  # 尝试获取图像嵌入
                            if embedding is not None:  # 如果成功获取嵌入
                                rel2vis[relation] = embedding  # 形状为 [1, 2304]

# 将实体和关系保存为文件
with open(os.path.join(output_dir, "entities.txt"), "w", encoding="utf-8") as f:
    for entity in entities:
        f.write(f"{entity}\n")

with open(os.path.join(output_dir, "relations.txt"), "w", encoding="utf-8") as f:
    for relation in relations:
        f.write(f"{relation}\n")

# 保存三元组（训练集）
with open(os.path.join(output_dir, "train.txt"), "w", encoding="utf-8") as f:
    for triplet in triplets:
        f.write(f"{triplet[0]}\t{triplet[1]}\t{triplet[2]}\n")

with open(os.path.join(output_dir, "test.txt"), "w", encoding="utf-8") as f:
    for triplet in triplets:
        f.write(f"{triplet[0]}\t{triplet[1]}\t{triplet[2]}\n")

# 划分训练集和验证集（假设我们随机选择 20% 的数据作为验证集）
random.shuffle(triplets)
split_idx = int(0.8 * len(triplets))  # 80% 作为训练集，20% 作为验证集
valid_triplets = triplets[split_idx:]

# 保存三元组（验证集）
with open(os.path.join(output_dir, "valid.txt"), "w", encoding="utf-8") as f:
    for triplet in valid_triplets:
        f.write(f"{triplet[0]}\t{triplet[1]}\t{triplet[2]}\n")

# 保存实体的文本特征
torch.save(ent2txt, os.path.join(output_dir, "textual_features_ent.pt"))

# 保存关系的视觉特征
torch.save(rel2vis, os.path.join(output_dir, "visual_features_rel.pt"))

# 检查形状
if rel2vis:
    print(rel2vis[list(rel2vis.keys())[0]].shape)  # 应该输出 torch.Size([1, 2304])

print(f"All files are saved to {output_dir}")