import torch
import torch.nn as nn

class VISTA(nn.Module):
    def __init__(self, num_ent, num_rel, ent_vis, rel_vis, dim_vis, ent_txt, rel_txt, dim_txt, ent_vis_mask, rel_vis_mask, \
                 dim_str, num_head, dim_hid, num_layer_enc_ent, num_layer_enc_rel, num_layer_dec, dropout = 0.1, \
                 emb_dropout = 0.6, vis_dropout = 0.1, txt_dropout = 0.1):
        super(VISTA, self).__init__()
        self.dim_str = dim_str
        self.num_head = num_head
        self.dim_hid = dim_hid
        self.num_ent = num_ent
        self.num_rel = num_rel

        # self.ent_vis = ent_vis
        self.rel_vis = rel_vis
        self.ent_txt = ent_txt.unsqueeze(dim = 1)
        # self.rel_txt = rel_txt.unsqueeze(dim = 1)

        false_ents = torch.full((self.num_ent,1),False).cuda()
        # self.ent_mask = torch.cat([false_ents, false_ents, ent_vis_mask, false_ents], dim = 1)
        self.ent_mask = torch.cat([false_ents, false_ents, false_ents], dim=1)

        false_rels = torch.full((self.num_rel,1),False).cuda()
        self.rel_mask = torch.cat([false_rels, rel_vis_mask, false_rels], dim = 1)

        self.ent_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.rel_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.ent_embeddings = nn.Parameter(torch.Tensor(num_ent, 1, dim_str))
        self.rel_embeddings = nn.Parameter(torch.Tensor(num_rel, 1 ,dim_str))
        self.lp_token = nn.Parameter(torch.Tensor(1,dim_str))

        self.str_ent_ln = nn.LayerNorm(dim_str)
        self.str_rel_ln = nn.LayerNorm(dim_str)
        self.vis_ln = nn.LayerNorm(dim_str)
        self.txt_ln = nn.LayerNorm(dim_str)

        self.embdr = nn.Dropout(p = emb_dropout)
        self.visdr = nn.Dropout(p = vis_dropout)
        self.txtdr = nn.Dropout(p = txt_dropout)


        self.pos_str_ent = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_vis_ent = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_txt_ent = nn.Parameter(torch.Tensor(1,1,dim_str))

        self.pos_str_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_vis_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_txt_rel = nn.Parameter(torch.Tensor(1,1,dim_str))

        self.pos_head = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_tail = nn.Parameter(torch.Tensor(1,1,dim_str))
        
        self.proj_ent_vis = nn.Linear(dim_vis, dim_str)
        self.proj_txt = nn.Linear(dim_txt, dim_str)

        self.proj_rel_vis = nn.Linear(dim_vis, dim_str)


        ent_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)
        self.ent_encoder = nn.TransformerEncoder(ent_encoder_layer, num_layer_enc_ent)
        
        rel_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)
        self.rel_encoder = nn.TransformerEncoder(rel_encoder_layer, num_layer_enc_rel)
        
        decoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layer_dec)

        self.init_weights()
        

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings)
        nn.init.xavier_uniform_(self.rel_embeddings)
        nn.init.xavier_uniform_(self.proj_ent_vis.weight)
        nn.init.xavier_uniform_(self.proj_rel_vis.weight)
        nn.init.xavier_uniform_(self.proj_txt.weight)

        nn.init.xavier_uniform_(self.ent_token)
        nn.init.xavier_uniform_(self.rel_token)
        nn.init.xavier_uniform_(self.lp_token)
        nn.init.xavier_uniform_(self.pos_str_ent)
        nn.init.xavier_uniform_(self.pos_vis_ent)
        nn.init.xavier_uniform_(self.pos_txt_ent)
        nn.init.xavier_uniform_(self.pos_str_rel)
        nn.init.xavier_uniform_(self.pos_vis_rel)
        nn.init.xavier_uniform_(self.pos_txt_rel)
        nn.init.xavier_uniform_(self.pos_head)
        nn.init.xavier_uniform_(self.pos_rel)
        nn.init.xavier_uniform_(self.pos_tail)

        self.proj_ent_vis.bias.data.zero_()
        self.proj_rel_vis.bias.data.zero_()
        self.proj_txt.bias.data.zero_()

    def forward(self):
        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)) + self.pos_str_ent
        # rep_ent_vis = self.visdr(self.vis_ln(self.proj_ent_vis(self.ent_vis))) + self.pos_vis_ent
        rep_ent_txt = self.txtdr(self.txt_ln(self.proj_txt(self.ent_txt))) + self.pos_txt_ent
        # ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_vis, rep_ent_txt], dim = 1)
        ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_txt], dim = 1)
        ent_embs = self.ent_encoder(ent_seq, src_key_padding_mask = self.ent_mask)[:,0]

        rel_tkn = self.rel_token.tile(self.num_rel, 1, 1)
        rep_rel_str = self.embdr(self.str_rel_ln(self.rel_embeddings)) + self.pos_str_rel
        # print("rel_vis.shape",self.rel_vis.shape)
        
        rep_rel_vis = self.visdr(self.vis_ln(self.proj_rel_vis(self.rel_vis))) + self.pos_vis_rel
        # rep_rel_txt = self.txtdr(self.txt_ln(self.proj_txt(self.rel_txt))) + self.pos_txt_rel
        # rel_seq = torch.cat([rel_tkn, rep_rel_str, rep_rel_vis, rep_rel_txt], dim = 1)
        rel_seq = torch.cat([rel_tkn, rep_rel_str, rep_rel_vis], dim = 1)
        rel_embs = self.rel_encoder(rel_seq, src_key_padding_mask = self.rel_mask)[:,0]

        return torch.cat([ent_embs, self.lp_token], dim = 0), rel_embs
    
    def score(self, emb_ent, emb_rel, triplets):
        h_seq = emb_ent[triplets[:,0] - self.num_rel].unsqueeze(dim = 1) + self.pos_head
        # print("h_seq.shape",h_seq.shape)
        # h_seq.shape torch.Size([1, 1, 256])

        r_seq = emb_rel[triplets[:,1] - self.num_ent].unsqueeze(dim = 1) + self.pos_rel
        # print("r_seq.shape",r_seq.shape)
        # r_seq.shape torch.Size([1, 1, 256])
        
        t_seq = emb_ent[triplets[:,2] - self.num_rel].unsqueeze(dim = 1) + self.pos_tail
        # print("t_seq.shape",t_seq.shape)

        dec_seq = torch.cat([h_seq, r_seq, t_seq], dim = 1)
        # print("dec_seq",dec_seq.shape)

        # print(self.decoder(dec_seq).shape)
        # torch.Size([1, 3, 256])

        output_dec = self.decoder(dec_seq)[triplets == self.num_ent + self.num_rel]

        score = torch.inner(output_dec, emb_ent[:-1])
        return score

    def predict_new_relation(self, tail_id, new_rel_features):
        """处理新关系的预测"""
        # 投影新关系特征
        print("开始进行预测")
        rel_vis = self.visdr(
            self.vis_ln(
                self.proj_rel_vis(new_rel_features)
            ) + self.pos_vis_rel)
        
        # 构建关系序列
        rel_tkn = self.rel_token.expand(1, -1, -1)
        rel_seq = torch.cat([rel_tkn, rel_vis], dim=1)
        rel_emb = self.rel_encoder(rel_seq)[:, 0]
        print("关系特征提取完毕",rel_emb.shape)
        
        # 计算得分
        ent_embs, _ = self.forward()
        
        masked_triple = torch.tensor([
            [self.num_ent + self.num_rel, -1, tail_id + self.num_rel]  # -1表示新关系
        ]).cuda()

        h_seq = ent_embs[masked_triple[:, 0] - self.num_rel].unsqueeze(dim = 1) + self.pos_head
        print("头实体特征提取完毕",h_seq.shape)
        
        r_seq = rel_emb.unsqueeze(dim = 1) + self.pos_rel
        print("关系特征提取完毕",r_seq.shape)
        
        t_seq = ent_embs[masked_triple[:, 2] - self.num_rel].unsqueeze(dim = 1) + self.pos_tail
        print("尾实体特征提取完毕",t_seq.shape)

        dec_seq = torch.cat([h_seq, r_seq, t_seq], dim = 1)

        print("dec_seq",dec_seq.shape)
        output_dec = self.decoder(dec_seq)[masked_triple == self.num_ent + self.num_rel]
        score = torch.inner(output_dec, ent_embs[:-1])

        return score