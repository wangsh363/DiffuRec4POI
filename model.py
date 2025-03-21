import torch.nn as nn
import torch
import math
from diffurec import DiffuRec
import torch.nn.functional as F
import copy
import numpy as np
from step_sample import LossAwareSampler
import torch as th


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Att_Diffuse_model(nn.Module):
    def __init__(self, diffu, args):
        super(Att_Diffuse_model, self).__init__()
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num+1
        # 这是一个嵌入层。第一个参数是最大索引值，第二个参数是嵌入层维度。用来给物品id编码
        # 最大索引值通过smap的长度来确定。
        # 但是ca的smap不是按照长度来分配的。要改一下。
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_dim)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.diffu = diffu
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_rec = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.MSELoss()

    def diffu_pre(self, item_rep, tag_emb, TimeStamp, mask_seq):
        seq_rep_diffu, item_rep_out, weights, t  = self.diffu(item_rep, tag_emb, TimeStamp, mask_seq)
        return seq_rep_diffu, item_rep_out, weights, t

    def reverse(self, item_rep, noise_x_t, TimeStamp, mask_seq):
        reverse_pre = self.diffu.reverse_p_sample(item_rep, noise_x_t, TimeStamp, mask_seq)
        return reverse_pre

    def loss_rec(self, scores, labels):
        return self.loss_ce(scores, labels.squeeze(-1))

    def loss_diffu(self, rep_diffu, labels):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        scores_pos = scores.gather(1 , labels)  ## labels: b x 1
        scores_neg_mean = (torch.sum(scores, dim=-1).unsqueeze(-1)-scores_pos)/(scores.shape[1]-1)
      
        loss = torch.min(-torch.log(torch.mean(torch.sigmoid((scores_pos - scores_neg_mean).squeeze(-1)))), torch.tensor(1e8))
       
        # if isinstance(self.diffu.schedule_sampler, LossAwareSampler):
        #     self.diffu.schedule_sampler.update_with_all_losses(t, loss.detach())
        # loss = (loss * weights).mean()
        return loss   

    def loss_diffu_ce(self, rep_diffu, labels):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        """
        ### norm scores
        item_emb_norm = F.normalize(self.item_embeddings.weight, dim=-1)
        rep_diffu_norm = F.normalize(rep_diffu, dim=-1)
        temperature = 0.07
        scores = torch.matmul(rep_diffu_norm, item_emb_norm.t())/temperature
        """
        return self.loss_ce(scores, labels.squeeze(-1))

    def diffu_rep_pre(self, rep_diffu):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        # 计算前后两个向量的相似度得分。后面这个weight好像是可学习的参数矩阵
        return scores
    
    def loss_rmse(self, rep_diffu, labels):
        rep_gt = self.item_embeddings(labels).squeeze(1)
        return torch.sqrt(self.loss_mse(rep_gt, rep_diffu))
    
    def routing_rep_pre(self, rep_diffu):
        item_norm = (self.item_embeddings.weight**2).sum(-1).view(-1, 1)  ## N x 1
        rep_norm = (rep_diffu**2).sum(-1).view(-1, 1)  ## B x 1
        sim = torch.matmul(rep_diffu, self.item_embeddings.weight.t())  ## B x N
        dist = rep_norm + item_norm.transpose(0, 1) - 2.0 * sim
        dist = torch.clamp(dist, 0.0, np.inf)
        
        return -dist

    def regularization_rep(self, seq_rep, mask_seq):
        seqs_norm = seq_rep/seq_rep.norm(dim=-1)[:, :, None]
        seqs_norm = seqs_norm * mask_seq.unsqueeze(-1)
        cos_mat = torch.matmul(seqs_norm, seqs_norm.transpose(1, 2))
        cos_sim = torch.mean(torch.mean(torch.sum(torch.sigmoid(-cos_mat), dim=-1), dim=-1), dim=-1)  ## not real mean
        return cos_sim

    def regularization_seq_item_rep(self, seq_rep, item_rep, mask_seq):
        item_norm = item_rep/item_rep.norm(dim=-1)[:, :, None]
        item_norm = item_norm * mask_seq.unsqueeze(-1)

        seq_rep_norm = seq_rep/seq_rep.norm(dim=-1)[:, None]
        sim_mat = torch.sigmoid(-torch.matmul(item_norm, seq_rep_norm.unsqueeze(-1)).squeeze(-1))
        return torch.mean(torch.sum(sim_mat, dim=-1)/torch.sum(mask_seq, dim=-1))

    # sequence是输入的序列，最后一个数据是时间，前面的是历史交互元组(物品，时间)。tag是label标签。
    # train_flag表示是否为训练模式
    def forward(self, sequence, tag, train_flag=True): 
        seq_length = sequence.size(1)   # 用户的历史行为序列（物品 ID 序列）
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        # position_embeddings = self.position_embeddings(position_ids)

        # 现在把sequence里的时间信息取出来
        # 理想的数据是这样的：
        # sequence的size为：torch.Size([512, 50, 2]), 也就是（batch_size，max_len, 2）
        # 这个时间戳的大小应该是torch.Size([512, 50])，id的大小也是[512,50]，即（batch_size，max_len）
        last_timestamp = sequence[..., 1]  # 取最后一维的第一个值
        sequence = sequence[..., 0]  # 取最后一维的第二个值

        # print("Max index:", sequence.max().item())  # 最大索引
        # print("Min index:", sequence.min().item())  # 最小索引
        # print("Embedding size:", self.item_embeddings.num_embeddings)  # 允许的最大索引

        item_embeddings = self.item_embeddings(sequence)  # 将离散的整数索引映射到连续的高维空间中
        item_embeddings = self.embed_dropout(item_embeddings)  ## dropout first than layernorm
        # item_embeddings是历史交互序列的嵌入

        # item_embeddings = item_embeddings + position_embeddings
        item_embeddings = self.LayerNorm(item_embeddings)  # 归一化
        
        mask_seq = (sequence>0).float()  # 这行代码的作用是生成一个掩码（mask），
        # 用于标识输入序列 sequence 中哪些位置是有效的（非零），哪些位置是无效的（填充值或零值）。
        # float是把布尔值转化为0和1
        
        if train_flag:  # 如果是训练模式
            tag_emb = self.item_embeddings(tag.squeeze(-1))  ## B x H   # 这个tag就是x0
            rep_diffu, rep_item, weights, t = self.diffu_pre(item_embeddings, tag_emb, last_timestamp, mask_seq)  # 进行扩散
            # 输入的分别是：历史交互序列的嵌入表示、tag(就是x0)、位置掩码
            # 输出的分别是：
            # rep_diffu：重建的x0_hat
            # rep_item:（h1,h2,...,hn）
            # weights是权重，t是时间步
            
            # item_rep_dis = self.regularization_rep(rep_item, mask_seq)
            # seq_rep_dis = self.regularization_seq_item_rep(rep_diffu, rep_item, mask_seq)
            
            item_rep_dis = None
            seq_rep_dis = None
        else:  # 如果是推理模式
            # noise_x_t = th.randn_like(tag_emb)
            noise_x_t = th.randn_like(item_embeddings[:,-1,:])
            rep_diffu = self.reverse(item_embeddings, noise_x_t, last_timestamp, mask_seq)
            weights, t, item_rep_dis, seq_rep_dis = None, None, None, None

        # item_rep = self.model_main(item_embeddings, rep_diffu, mask_seq)
        # seq_rep = item_rep[:, -1, :]
        # scores = torch.matmul(seq_rep, self.item_embeddings.weight.t())
        scores = None
        return scores, rep_diffu, weights, t, item_rep_dis, seq_rep_dis
        

def create_model_diffu(args):
    diffu_pre = DiffuRec(args)
    return diffu_pre

