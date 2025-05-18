import torch
from torch.utils.data import Dataset
import random
import os
from tqdm import tqdm

class VTKG(Dataset):
    def __init__(self, data, logger, max_vis_len = -1):
        self.data = data
        self.logger = logger
        self.dir = f"data/{data}/"
        self.ent2id = {}
        self.id2ent = []
        self.rel2id = {}
        self.id2rel = []
        with open(self.dir + "entities.txt") as f:
            for idx, line in enumerate(f.readlines()):
                self.ent2id[line.strip()] = idx
                self.id2ent.append(line.strip())
        self.num_ent = len(self.ent2id)

        with open(self.dir + "relations.txt") as f:
            for idx, line in enumerate(f.readlines()):
                self.rel2id[line.strip()] = idx
                self.id2rel.append(line.strip())
        self.num_rel = len(self.rel2id)

        self.train = []
        with open(self.dir + "train.txt") as f:
            for line in f.readlines():
                h,r,t = line.strip().split("\t")
                self.train.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.valid = []
        with open(self.dir + "valid.txt") as f:
            for line in f.readlines():
                h,r,t = line.strip().split("\t")
                self.valid.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.test = []
        with open(self.dir + "test.txt") as f:
            for line in f.readlines():
                h,r,t = line.strip().split("\t")
                self.test.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))
        
        self.filter_dict = {}

        for data_split in [self.train, self.valid, self.test]:
            for triplet in data_split:
                h,r,t = triplet
                if (-1, r, t) not in self.filter_dict:
                    self.filter_dict[(-1,r,t)] = []
                self.filter_dict[(-1,r,t)].append(h)
                if (h, r, -1) not in self.filter_dict:
                    self.filter_dict[(h,r,-1)] = []
                self.filter_dict[(h,r,-1)].append(t)

        self.max_vis_len_ent = max_vis_len
        self.max_vis_len_rel = max_vis_len
        self.gather_vis_feature()
        self.gather_txt_feature()
    
    def sort_vis_features(self, item='entity'):
        if item == 'entity':
            vis_feats = torch.load(self.dir + 'visual_features_ent.pt')
        elif item == 'relation':
            vis_feats = torch.load(self.dir + 'visual_features_rel.pt',weights_only=True)
        else:
            raise NotImplementedError
        
        # Get the current device (either 'cuda' if a GPU is available or 'cpu')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        sorted_vis_feats = {}
        for obj in tqdm(vis_feats):
            if item == 'entity' and obj not in self.ent2id:
                continue
            if item == 'relation' and obj not in self.rel2id:
                continue
            
            # Move visual features to the same device
            num_feats = len(vis_feats[obj])
            sim_val = torch.zeros(num_feats).to(device)  # Ensure sim_val is on the correct device
            
            # Move the features to the same device
            cudaed_feats = vis_feats[obj].to(device)  # Make sure the visual features are on the correct device
            
            # Calculate the similarity and sort the features
            for i in tqdm(range(num_feats)):
                sims = torch.inner(cudaed_feats[i], cudaed_feats[i:])
                sim_val[i:] += sims
                sim_val[i] += sims.sum() - torch.inner(cudaed_feats[i], cudaed_feats[i])
            
            # Sort the features based on similarity values
            sorted_vis_feats[obj] = vis_feats[obj].to(device)[torch.argsort(sim_val, descending=True)]

        # Save the sorted visual features
        if item == 'entity':
            torch.save(sorted_vis_feats, self.dir + "visual_features_ent_sorted.pt")
        else:
            torch.save(sorted_vis_feats, self.dir + "visual_features_rel_sorted.pt")
        
        return sorted_vis_feats


    def gather_vis_feature(self):
        # 在方法开始处定义device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if os.path.isfile(self.dir + 'visual_features_rel_sorted.pt'):
            self.rel2vis = torch.load(self.dir + 'visual_features_rel_sorted.pt', 
                                     map_location=device,
                                     weights_only=True)
        elif os.path.isfile(self.dir + 'visual_features_rel.pt'):
            self.rel2vis = self.sort_vis_features(item = 'relation')
        else:
            self.rel2vis = {}
        
        # print(self.rel2vis[list(self.rel2vis.keys())[0]].shape)
        # print(self.rel2vis[list(self.rel2vis.keys())[0]].shape)
        # self.vis_feat_size = len(self.ent2vis[list(self.ent2vis.keys())[0]][0])
        self.vis_feat_size = len(self.rel2vis[list(self.rel2vis.keys())[0]][0])
        # self.vis_feat_size = len(self.rel2vis[list(self.rel2vis.keys())[0]])
        # print(self.vis_feat_size)

        total_num = 0
        if self.max_vis_len_ent != -1:
            # for ent_name in self.ent2vis:
            #     num_feats = len(self.ent2vis[ent_name])
            #     total_num += num_feats
            #     self.ent2vis[ent_name] = self.ent2vis[ent_name][:self.max_vis_len_ent]
            for rel_name in self.rel2vis:
                self.rel2vis[rel_name] = self.rel2vis[rel_name][:self.max_vis_len_rel]
        else:
            # for ent_name in self.ent2vis:
            #     num_feats = len(self.ent2vis[ent_name])
            #     total_num += num_feats
            #     if self.max_vis_len_ent < len(self.ent2vis[ent_name]):
            #         self.max_vis_len_ent = len(self.ent2vis[ent_name])
            # self.max_vis_len_ent = max(self.max_vis_len_ent, 0)
            for rel_name in self.rel2vis:
                if self.max_vis_len_rel < len(self.rel2vis[rel_name]):
                    self.max_vis_len_rel = len(self.rel2vis[rel_name])
            self.max_vis_len_rel = max(self.max_vis_len_rel, 0)
        
        # self.ent_vis_mask = torch.full((self.num_ent, self.max_vis_len_ent), True).cuda()
        # self.ent_vis_matrix = torch.zeros((self.num_ent, self.max_vis_len_ent, self.vis_feat_size)).cuda()
        self.rel_vis_mask = torch.full((self.num_rel, self.max_vis_len_rel), True).to(device)
        self.rel_vis_matrix = torch.zeros((self.num_rel, self.max_vis_len_rel, self.vis_feat_size)).to(device)


        
        # for ent_name in self.ent2vis:
        #     ent_id = self.ent2id[ent_name]
        #     num_feats = len(self.ent2vis[ent_name])
        #     self.ent_vis_mask[ent_id, :num_feats] = False
        #     self.ent_vis_matrix[ent_id, :num_feats] = self.ent2vis[ent_name]

        for rel_name in self.rel2vis:
            rel_id = self.rel2id[rel_name]
            num_feats = len(self.rel2vis[rel_name])
            self.rel_vis_mask[rel_id, :num_feats] = False
            self.rel_vis_matrix[rel_id, :num_feats] = self.rel2vis[rel_name]
        
        print("rel_vis_matrix.shape",self.rel_vis_matrix.shape)

    def gather_txt_feature(self):
        # 定义device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ent2txt = torch.load(self.dir + 'textual_features_ent.pt',
                                 map_location=device,
                                 weights_only=True)
        self.txt_feat_size = len(self.ent2txt[self.id2ent[0]])

        self.ent_txt_matrix = torch.zeros((self.num_ent, self.txt_feat_size)).to(device)

        for ent_name in self.ent2id:
            self.ent_txt_matrix[self.ent2id[ent_name]] = self.ent2txt[ent_name]

        # for rel_name in self.rel2id:
        #     self.rel_txt_matrix[self.rel2id[rel_name]] = self.rel2txt[rel_name]


    def __len__(self):
        return len(self.train)
    
    def __getitem__(self, idx):
        h,r,t = self.train[idx]
        if random.random() < 0.5:
            masked_triplet = [self.num_ent + self.num_rel, r + self.num_ent, t + self.num_rel]
            label = h
        else:
            masked_triplet = [h + self.num_rel, r + self.num_ent, self.num_ent + self.num_rel]
            label = t
        
        return torch.tensor(masked_triplet), torch.tensor(label)
    
    def add_new_relation(self, rel_name, image_features):
        """动态添加新关系到知识图谱"""
        # 分配新ID（使用负数避免冲突）
        new_rel_id = -(len(self.rel2id) + 1)
        self.rel2id[rel_name] = new_rel_id
        self.id2rel.append(rel_name)
        
        # 确保特征维度一致
        num_images = image_features.shape[0]  # 新关系的图像数量
        if num_images < self.max_vis_len_rel:
            # 如果图像数量不足，用零填充
            padding = torch.zeros(
                (self.max_vis_len_rel - num_images, image_features.shape[1]),
                device=image_features.device
            )
            processed_feats = torch.cat([image_features, padding], dim=0)
        else:
            # 如果图像数量超过最大值，截断
            processed_feats = image_features[:self.max_vis_len_rel]
        
        # 检查特征形状
        assert processed_feats.shape == (self.max_vis_len_rel, 2304), f"特征形状不匹配: {processed_feats.shape}"
        
        # 扩展视觉特征矩阵
        self.rel_vis_matrix = torch.cat([
            self.rel_vis_matrix,
            processed_feats.unsqueeze(0).cuda()  # 添加 batch 维度
        ], dim=0)
        
        # 更新掩码
        new_mask = torch.full((1, self.max_vis_len_rel), True).cuda()
        new_mask[0, :num_images] = False  # 仅对实际图像部分设置为 False
        self.rel_vis_mask = torch.cat([self.rel_vis_mask, new_mask], dim=0)

    def _process_features(self, features):
        """确保特征与训练时格式一致"""
        # 排序逻辑（与训练时相同）
        sim_val = torch.zeros(len(features))
        for i in range(len(features)):
            sims = torch.inner(features[i], features)
            sim_val += sims
        return features[torch.argsort(sim_val, descending=True)]
