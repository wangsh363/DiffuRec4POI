import torch.nn as nn
import torch
import math

from torch.nn.functional import dropout

from diffurec import DiffuRec, tile_pre
import torch.nn.functional as F
import copy
import numpy as np
from step_sample import LossAwareSampler
import torch as th
from torch.nn import TransformerEncoderLayer, TransformerEncoder  # Transformer编码器相关组件

class ArcFaceLoss(nn.Module):
    def __init__(self, margin=0.1, scale=64):
        super(ArcFaceLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.mm = self.sin_m * self.margin
        self.threshold = math.cos(math.pi - self.margin)

    def forward(self, x, c, labels, c_mask=None):
        labels = labels.view(-1).long()
        num_classes = c.shape[0]
        if labels.max() >= num_classes or labels.min() < 0:
            print(f"警告: 无效 labels 检测到，min={labels.min().item()}, max={labels.max().item()}, num_classes={num_classes}")
            labels = torch.clamp(labels, 0, num_classes - 1)
        x = F.normalize(x)
        c = F.normalize(c, p=2, dim=-1)
        cos_theta = torch.matmul(x.unsqueeze(1), c.transpose(-2, -1)).squeeze(1)
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2) + 1e-7)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(cos_theta > self.threshold, cos_theta_m, cos_theta - self.mm)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * cos_theta_m) + ((1.0 - one_hot) * cos_theta)
        output *= self.scale
        loss = F.cross_entropy(output, labels)
        return loss

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

class embedding(nn.Module):
    """
    自定义嵌入模块，用于将离散索引转换为连续的嵌入向量
    """
    def __init__(self, vocab_size, num_units, zeros_pad=True, scale=True):
        """
        初始化嵌入层
        参数:
            vocab_size: 词汇表大小（整数），表示输入的离散索引数量
            num_units: 嵌入维度（整数），表示每个索引映射到的向量长度
            zeros_pad: 布尔值，若为True，则将索引0的嵌入向量初始化为全零（通常用于填充）
            scale: 布尔值，若为True，则输出嵌入向量会乘以嵌入维度的平方根（用于归一化）
        """
        super(embedding, self).__init__()  # 调用父类nn.Module的初始化方法
        self.vocab_size = vocab_size  # 保存词汇表大小
        self.num_units = num_units  # 保存嵌入维度
        self.zeros_pad = zeros_pad  # 保存是否填充零的标志
        self.scale = scale  # 保存是否缩放的标志
        # 定义嵌入查找表，作为可训练参数，形状为 [vocab_size, num_units]
        self.lookup_table = nn.Parameter(torch.Tensor(vocab_size, num_units))
        # 使用Xavier正态分布初始化嵌入表
        nn.init.xavier_normal_(self.lookup_table.data)
        if self.zeros_pad:  # 如果需要填充零
            self.lookup_table.data[0, :].fill_(0)  # 将索引0的嵌入向量设为全零

    def forward(self, inputs):
        """
        前向传播，将输入索引转换为嵌入向量
        参数:
            inputs: 输入张量，包含离散索引，形状可以是任意（如 [N, L]）
        返回:
            outputs: 嵌入向量张量，形状为 [N, L, num_units]
        """
        if self.zeros_pad:  # 如果启用零填充
            self.padding_idx = 0  # 设置填充索引为0
        else:
            self.padding_idx = -1  # 否则设置为-1（不填充）

        # 使用torch.nn.functional.embedding函数查找嵌入向量
        outputs = F.embedding(
            inputs,  # 输入索引张量
            self.lookup_table,  # 嵌入查找表
            self.padding_idx,  # 填充索引
            None,  # max_norm，未使用
            2,  # norm_type，未使用
            False,  # scale_grad_by_freq，未使用
            False  # sparse，未使用
        )  # 参数设置参考torch.nn.modules.sparse.Embedding

        if self.scale:  # 如果需要缩放
            outputs = outputs * (self.num_units ** 0.5)  # 乘以嵌入维度的平方根

        return outputs  # 返回嵌入结果

class PositionalEmbedding(nn.Module):
    """
    位置嵌入模块（可学习的位置编码）
    """

    def __init__(self, d_model, dropout=0.1, max_len=120):
        """
        初始化位置嵌入
        参数:
            d_model: 模型维度
            dropout: Dropout概率，默认为0.1
            max_len: 最大序列长度，默认为120
        """
        super(PositionalEmbedding, self).__init__()
        # 定义可学习的位置嵌入表
        self.pos_emb_table = embedding(max_len, d_model, zeros_pad=False, scale=False)
        pos_vector = torch.arange(max_len)  # 生成位置索引 [0, 1, ..., max_len-1]
        self.dropout = nn.Dropout(p=dropout)  # 定义Dropout层
        self.register_buffer('pos_vector', pos_vector)  # 注册位置向量为缓冲区

    def forward(self, x):
        """
        前向传播，添加位置嵌入
        参数:
            x: 输入张量，形状 [L, N, d_model]
        返回:
            添加位置嵌入并应用Dropout后的张量
        """
        # 获取位置嵌入并扩展到批次维度
        pos_emb = self.pos_emb_table(self.pos_vector[:x.size(0)].unsqueeze(1).repeat(1, x.size(1)))
        x += pos_emb  # 将位置嵌入加到输入上
        return self.dropout(x)  # 应用Dropout


class TilePosEnc(nn.Module):
    def __init__(self, d_model, device='cuda'):
        super(TilePosEnc, self).__init__()
        self.d_model = d_model
        self.device = device

    def forward(self, tile_embeds, coords):
        """
        为tile嵌入添加基于POI真实经纬度的正余弦位置编码
        参数:
            tile_embeds: tile嵌入张量，形状 [batch_size, seq_len, d_model]
            coords: POI的经纬度序列，形状 [batch_size, seq_len, 2]，最后一维为 [lat, lon]
        返回:
            添加位置编码后的tile嵌入，形状 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = tile_embeds.shape
        lat = coords[:, :, 0].unsqueeze(-1)  # [batch_size, seq_len, 1]
        lon = coords[:, :, 1].unsqueeze(-1)  # [batch_size, seq_len, 1]

        # 标准化经纬度到 [0, 1],这个标准化是原论文不带的
        lat = (lat + 90) / 180  # 纬度从 [-90, 90] 映射到 [0, 1]
        lon = (lon + 180) / 360  # 经度从 [-180, 180] 映射到 [0, 1]

        div_term = torch.exp(
            torch.arange(0, self.d_model // 2, 2, device=self.device) * (-math.log(10000.0) / (self.d_model // 2))
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, d_model//4]

        pe = torch.zeros(batch_size, seq_len, self.d_model, device=self.device)
        pe[:, :, 0:self.d_model // 2:2] = torch.sin(lat * div_term)
        pe[:, :, 1:self.d_model // 2:2] = torch.cos(lat * div_term)
        pe[:, :, self.d_model // 2::2] = torch.sin(lon * div_term)
        pe[:, :, self.d_model // 2 + 1::2] = torch.cos(lon * div_term)

        return tile_embeds + pe

class Att_Diffuse_model(nn.Module):
    def __init__(self, diffu, args, quadkey_vocab_size, tile_vocab_size, tile_to_poi, poi_to_tile):
        super(Att_Diffuse_model, self).__init__()
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num
        self.batch_size = args.batch_size
        self.num_gpu = args.num_gpu
        self.user_num = args.user_num
        # 这是一个嵌入层。第一个参数是最大索引值，第二个参数是嵌入层维度。用来给物品id编码
        # 最大索引值通过smap的长度来确定。
        # 但是ca的smap不是按照长度来分配的。要改一下。
        self.PAD_IDX = 0
        self.item_embeddings = nn.Embedding(self.item_num + 1, self.emb_dim, padding_idx=self.PAD_IDX)
        self.user_embeddings = nn.Embedding(self.user_num + 1, self.emb_dim, padding_idx=self.PAD_IDX)
        # quadkey嵌入
        self.quadkey_embeddings = nn.Embedding(quadkey_vocab_size, self.emb_dim)
        self.tile_embeddings = nn.Embedding(tile_vocab_size, self.emb_dim, padding_idx=self.PAD_IDX)
        self.tile_to_poi = tile_to_poi  # 存储瓦片到POI的映射

        self.poi_to_tile_tensor = torch.full((self.item_num + 1,), fill_value=-1, dtype=torch.long)
        for poi_idx, tile_idx in poi_to_tile.items():
            self.poi_to_tile_tensor[poi_idx] = tile_idx

        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)

        self.top_k_tiles = args.top_k_tiles
        self.tile_vocab_size = tile_vocab_size
        self.top_k_pois = args.top_k_pois
        self.arcface_loss_tile = ArcFaceLoss(margin=0.05, scale=64)  # 瓦片预测边距较小
        self.arcface_loss_poi = ArcFaceLoss(margin=0.1, scale=64)  # POI预测边距较大

        # 向quadkey添加自注意力机制
        self.quadkey_enc_layer = TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=args.nhead,  # args.nhead
            dim_feedforward=self.emb_dim,  # 前馈网络维度
            dropout=0.5,
            activation='gelu',
            batch_first=True
        )
        self.quadkey_encoder = TransformerEncoder(
            self.quadkey_enc_layer,
            num_layers=args.num_layers
        )
        self.quadkey_pos_encoder = PositionalEmbedding(self.emb_dim, dropout=0.5, max_len=12)

        # 分别创建两个扩散模型
        self.diffu_tile = tile_pre(args)  # 用于瓦片序列
        self.diffu_poi = DiffuRec(args)  # 用于POI序列

        self.diffu = diffu
        self.loss_ce = nn.CrossEntropyLoss()  # 交叉熵损失
        self.loss_ce_rec = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.MSELoss()

        self.tile_pos_enc = TilePosEnc(self.emb_dim, device=args.device)

        self.tile_temp_log = nn.Parameter(torch.tensor(0.0))  # log(temp)
        self.poi_temp_log = nn.Parameter(torch.tensor(0.0))
        # 初始化 <unk> 嵌入,避免与填充向量（全零）混淆。
        # with torch.no_grad():
        #     self.item_embeddings.weight[args.item_num - 1].normal_(mean=0, std=0.1)  # <unk> POI 嵌入
        #     self.tile_embeddings.weight[tile_vocab_size - 1].normal_(mean=0, std=0.1)  # <unk> 瓦片嵌入

    def diffu_pre(self, rep, tag_emb, timestamps, user_embeds, quadkey_rep, tile_rep_diffu, mask_seq):
        seq_rep_diffu, item_rep_out, weights, t, time_target, condition = self.diffu_poi(
            rep, tag_emb, timestamps, user_embeds, quadkey_rep, tile_rep_diffu, mask_seq
        )
        return seq_rep_diffu, item_rep_out, weights, t, time_target, condition

    def reverse(self, rep, noise_x_t, timestamps, user_embeds, quadkey_rep, mask_seq):
        reverse_pre, time_target = self.diffu_poi.reverse_p_sample(
            rep, noise_x_t, timestamps, user_embeds, quadkey_rep, mask_seq
        )
        return reverse_pre, time_target

    def loss_rec(self, scores, labels):
        return self.loss_ce(scores, labels.squeeze(-1))

    def loss_diffu(self, rep_diffu, labels):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        scores_pos = scores.gather(1, labels)  ## labels: b x 1
        scores_neg_mean = (torch.sum(scores, dim=-1).unsqueeze(-1) - scores_pos) / (scores.shape[1] - 1)

        loss = torch.min(-torch.log(torch.mean(torch.sigmoid((scores_pos - scores_neg_mean).squeeze(-1)))),
                         torch.tensor(1e8))
        return loss

    def loss_diffu_ce(self, rep_diffu, labels):

        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())  # 对self.item_embeddings.weight.t() 进行时间的改变
        # print("你好")
        # print(scores)
        # print(scores.size())
        # print("!!!!!!!!!!!!!!!!!!!")
        # print(labels.squeeze(-1))
        """
        ### norm scores
        item_emb_norm = F.normalize(self.item_embeddings.weight, dim=-1)
        rep_diffu_norm = F.normalize(rep_diffu, dim=-1)
        temperature = 0.07
        scores = torch.matmul(rep_diffu_norm, item_emb_norm.t())/temperature
        """
        return self.loss_ce(scores, labels.squeeze(-1))  # 作用是去掉最后一个维度

    def diffu_rep_pre(self, rep_diffu):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())  # 计算rep_difffu与所有物品的相似度，也就是每个物品的匹配分数
        # 计算前后两个向量的相似度得分。后面这个weight好像是可学习的参数矩阵
        return scores

    def loss_rmse(self, rep_diffu, labels):
        rep_gt = self.item_embeddings(labels).squeeze(1)
        return torch.sqrt(self.loss_mse(rep_gt, rep_diffu))

    def routing_rep_pre(self, rep_diffu):
        item_norm = (self.item_embeddings.weight ** 2).sum(-1).view(-1, 1)  ## N x 1
        rep_norm = (rep_diffu ** 2).sum(-1).view(-1, 1)  ## B x 1
        sim = torch.matmul(rep_diffu, self.item_embeddings.weight.t())  ## B x N
        dist = rep_norm + item_norm.transpose(0, 1) - 2.0 * sim
        dist = torch.clamp(dist, 0.0, np.inf)

        return -dist

    def regularization_rep(self, seq_rep, mask_seq):
        seqs_norm = seq_rep / seq_rep.norm(dim=-1)[:, :, None]
        seqs_norm = seqs_norm * mask_seq.unsqueeze(-1)
        cos_mat = torch.matmul(seqs_norm, seqs_norm.transpose(1, 2))
        cos_sim = torch.mean(torch.mean(torch.sum(torch.sigmoid(-cos_mat), dim=-1), dim=-1), dim=-1)  ## not real mean
        return cos_sim

    def regularization_seq_item_rep(self, seq_rep, item_rep, mask_seq):
        item_norm = item_rep / item_rep.norm(dim=-1)[:, :, None]
        item_norm = item_norm * mask_seq.unsqueeze(-1)

        seq_rep_norm = seq_rep / seq_rep.norm(dim=-1)[:, None]
        sim_mat = torch.sigmoid(-torch.matmul(item_norm, seq_rep_norm.unsqueeze(-1)).squeeze(-1))
        return torch.mean(torch.sum(sim_mat, dim=-1) / torch.sum(mask_seq, dim=-1))

    # 2stage损失计算方式,由于labels记录的是POI的，计算tiles损失时需要对应到POI上计算
    def loss_arcface(self, rep_diffu, labels, target_type="poi"):
        if target_type == "tile":
            embeddings = self.tile_embeddings.weight
            arcface_loss = self.arcface_loss_tile
        else:
            embeddings = self.item_embeddings.weight
            arcface_loss = self.arcface_loss_poi
        return arcface_loss(rep_diffu, embeddings, labels)
    
    def loss_two_stage(self, rep_tile, rep_poi, tile_labels, poi_labels, alpha=0):
        # (1) Tile-level scores and loss
        tile_scores = torch.matmul(rep_tile, self.tile_embeddings.weight.t())  # [B, T]
        tile_loss = self.loss_ce(tile_scores, tile_labels.squeeze(-1))

        # (2) POI-level scores
        poi_scores = torch.matmul(rep_poi, self.item_embeddings.weight.t())  # [B, P]

        # (3) 使用提前构造好的 self.poi_to_tile_tensor
        # [num_pois] -> [1, num_pois] -> [B, num_pois]
        poi_tile_indices = self.poi_to_tile_tensor.to(rep_poi.device).unsqueeze(0).expand(rep_poi.size(0), -1)  # [B, P]

        # [B, T] detach，防止 tile 分支反向传播影响
        tile_scores_detached = tile_scores.detach()
        # tile_scores_detached = tile_scores

        # poi_tile_scores: [B, P]，每个 poi 的 tile 得分（广播 + gather）
        poi_tile_scores = torch.gather(tile_scores_detached, dim=1, index=poi_tile_indices)

        # (4) 加权融合 POI 得分
        final_poi_scores = poi_scores + alpha * poi_tile_scores

        # (5) POI 层级的交叉熵损失
        poi_loss = self.loss_ce(final_poi_scores, poi_labels.squeeze(-1))

        # (6) 总损失
        return tile_loss, poi_loss, tile_scores, poi_scores, final_poi_scores
    

    def loss_two_stage_prob(self, rep_tile, rep_poi, tile_labels, poi_labels):
        # 1. logits
        tile_logits = torch.matmul(rep_tile, self.tile_embeddings.weight.t())  # [B, T]
        poi_logits = torch.matmul(rep_poi, self.item_embeddings.weight.t())    # [B, P]

        # 2. softmax with temperature
        # tile_temp = torch.exp(self.tile_temp_log)  # 确保 temp > 0
        # poi_temp = torch.exp(self.poi_temp_log)
        tile_temp = 7
        poi_temp = 4
        tile_probs = torch.softmax(tile_logits / tile_temp, dim=1)             # [B, T]
        poi_probs = torch.softmax(poi_logits / poi_temp, dim=1)                # [B, P]
        # print("tile_temp:", tile_temp.item(), "poi_temp:", poi_temp.item())
        # tile_probs = tile_probs.detach()

        # 3. gather tile_probs for each POI
        poi_tile_indices = self.poi_to_tile_tensor.to(rep_poi.device).unsqueeze(0).expand(rep_poi.size(0), -1)
        tile_probs_for_pois = torch.gather(tile_probs, dim=1, index=poi_tile_indices)  # [B, P]

        # 4. compute joint probs
        final_probs = tile_probs_for_pois * poi_probs                          # [B, P]
        final_probs = final_probs / final_probs.sum(dim=1, keepdim=True)      # optional normalize

        # 5. compute POI-level loss (joint)
        target = poi_labels.squeeze(-1).unsqueeze(1)                           # [B, 1]
        gathered_probs = torch.gather(final_probs, dim=1, index=target)       # [B, 1]
        joint_loss = -torch.log(gathered_probs + 1e-12).mean()

        # 6. compute tile-level loss
        tile_loss = self.loss_ce(tile_logits, tile_labels.squeeze(-1))        # [B]

        return tile_loss, joint_loss, tile_probs, poi_probs, final_probs


    # sequence是输入的序列，最后一个数据是[0, 时间]，前面的是历史交互元组(物品，时间)。tag是label标签。
    # train_flag表示是否为训练模式
    def forward(self, sequence, labels, tile_labels, train_flag=True, coords=None):
        # seq_length = sequence.size(1)   # 用户的历史行为序列（物品 ID 序列）
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        # position_embeddings = self.position_embeddings(position_ids)

        items, timestamps, uids, quadkeys, tiles = sequence
        # unk_tile_id = self.tile_vocab_size - 1  # <unk> 瓦片ID
        # unk_poi_id = self.item_num - 1  # <unk> POI ID

        # print("tile_vocab_size:", self.tile_vocab_size)
        # print("item_num:", self.item_num)
        # if tiles.max().item() >= self.tile_vocab_size or tiles.min().item() < 0:
        #     print(f"检测到无效瓦片 ID: min={tiles.min().item()}, max={tiles.max().item()}, 词汇表大小={self.tile_vocab_size}")
        #     tiles = torch.clamp(tiles, min=0, max=unk_tile_id)  # 映射到 <unk>
        # if items.max().item() >= self.item_num or items.min().item() < 0:
        #     print(f"检测到无效 items: min={items.min().item()}, max={items.max().item()}, item_num={self.item_num}")
        #     items = torch.clamp(items, min=0, max=unk_poi_id)  # 映射到 <unk>
        # 现在把sequence里的时间信息取出来
        # 理想的数据是这样的：
        # sequence为tuple3, 0是items([512, 50]), 1是timestamps([512, 50])， 2是quadkeys([512, 50, 12])

        item_embeddings_origin = self.item_embeddings(items)  # 将离散的整数索引映射到连续的高维空间中
        item_embeddings = self.embed_dropout(item_embeddings_origin)  ## dropout first than layernorm
        # item_embeddings是历史交互序列的嵌入

        # item_embeddings = item_embeddings + position_embeddings
        item_embeddings = self.LayerNorm(item_embeddings)  # 归一化

        # quadkey嵌入及自注意力
        # quadkey_embeds = self.quadkey_embeddings(quadkeys)  # [batch_size, seq_len, max_ngram_len, emb_dim]
        # quadkey_embeds = quadkey_embeds.view(quadkey_embeds.size(0)*quadkey_embeds.size(1), quadkey_embeds.size(2), quadkey_embeds.size(3)).permute(1, 0, 2)
        # # 添加位置编码,捕捉 n-gram token 在 Quadkey 序列（单个字符串）中的相对位置信息，Transformer模型对输入序列的顺序不敏感
        # quadkey_embeds = self.quadkey_pos_encoder(quadkey_embeds)
        # # Transformer 的自注意力机制可以捕捉 token 之间的复杂依赖关系
        # quadkey_embeds = self.quadkey_encoder(quadkey_embeds)
        # quadkey_embeds = torch.mean(quadkey_embeds, dim=0)  # 均值池化
        # quadkey_embeds = quadkey_embeds.view(item_embeddings_origin.size(0), item_embeddings_origin.size(1), quadkey_embeds.size(1))  # 重塑为 [batch_size, seq_len, emb_dim]
        quadkey_embeds = None

        poi_embeds = item_embeddings
        poi_embeds[:, -1] = 0.0
        # 用户序列嵌入和编码
        user_embeds = self.user_embeddings(uids)
        user_embeds = self.embed_dropout(user_embeds)  ## dropout first than layernorm
        user_embeds = self.LayerNorm(user_embeds)  # 归一化
        # 瓦片序列嵌入和编码
        tile_embeds = self.tile_embeddings(tiles)
        tile_embeds = self.embed_dropout(tile_embeds)  ## dropout first than layernorm
        tile_embeds = self.LayerNorm(tile_embeds)  # 归一化
        tile_embeds = self.tile_pos_enc(tile_embeds, coords)
        tile_embeds[:, -1] = 0.0
        # 问题就在如何去产生tile的位置序列,此外还需要看看位置编码层的初始化，利用经纬度的二维坐标生成后面继续判断两种类别，区分tile和pos的嵌入


        # 这个掩码需不需要修改，不是对于item使用了，对象变成了item_rep
        # mask_seq的大小是[512, 50]
        mask_seq = (items > 0).float()  # 这行代码的作用是生成一个掩码（mask），
        mask_seq_tile = (tiles > 0).float()
        # 用于标识输入序列 sequence 中哪些位置是有效的（非零），哪些位置是无效的（填充值或零值）。float是把布尔值转化为0和1
        # 有一个关键的参数：最后一个值一定要是有效的，因为最后一个值是由目标时间和0组成的。
        mask_seq[:, -1] = 1
        mask_seq_tile[:, -1] = 0

        if train_flag:
            labels_emb = self.item_embeddings(labels.squeeze(-1))
            tiles_emb = self.tile_embeddings(tile_labels.squeeze(-1))
            tile_rep_diffu = self.diffu_tile(
                tile_embeds, timestamps, user_embeds, quadkey_embeds, mask_seq_tile
            )
            poi_rep_diffu, poi_rep_item, poi_weights, poi_t, poi_time_target, condition = self.diffu_pre(
                poi_embeds, labels_emb, timestamps, user_embeds, quadkey_embeds, tile_rep_diffu, mask_seq
            )
            return condition, (tile_rep_diffu, poi_rep_diffu), (None, poi_weights), (None, poi_t), None, None, (
                None, poi_time_target)
        else:
            # 推理模式：分别去噪
            noise_x_t_tile = th.randn_like(item_embeddings[:, -1, :])
            noise_x_t_poi = th.randn_like(item_embeddings[:, -1, :])
            ######### 这个噪声是一样的吗，需不需要修改
            tile_rep_diffu = self.diffu_tile(
                tile_embeds, timestamps, user_embeds, quadkey_embeds, mask_seq_tile
            )
            poi_rep_diffu, poi_time_target = self.reverse(
                poi_embeds, noise_x_t_poi, timestamps, user_embeds, quadkey_embeds, mask_seq
            )

            # # 瓦片排序：生成Tile Ranking List
            # tile_scores = torch.matmul(tile_rep_diffu, self.tile_embeddings.weight.t())
            # _, top_k_tiles = torch.topk(tile_scores, k=self.top_k_tiles, dim=-1)  # Top K瓦片

            # # 从 Top-K 瓦片中提取候选 POI，并计算权重
            # batch_size = top_k_tiles.size(0)
            # candidate_pois = []
            # poi_weights = []  # 记录每个候选 POI 的权重
            # for i in range(batch_size):
            #     tile_ids = top_k_tiles[i].cpu().numpy()
            #     poi_set = set()
            #     poi_weight_dict = {}
            #     for rank, tile_id in enumerate(tile_ids):
            #         weight = 1.0 / (rank / 20 + 1)
            #         # weight = 1.0
            #         # if tile_id == unk_tile_id:
            #         #     continue  # 跳过 <unk> 瓦片
            #         if tile_id in self.tile_to_poi and self.tile_to_poi[tile_id]:
            #             for poi_id in self.tile_to_poi[tile_id]:
            #                 if poi_id >= self.item_num + 1 or poi_id < 0:
            #                     print(f"警告: 无效 POI ID {poi_id} 在瓦片 {tile_id}，跳过")
            #                     continue
            #                 poi_set.add(poi_id)
            #                 poi_weight_dict[poi_id] = poi_weight_dict.get(poi_id, 0.0) + weight
            #     candidate_pois.append(list(poi_set))
            #     poi_weights.append([poi_weight_dict.get(poi_id, 0.0) for poi_id in poi_set])

            # # 生成候选 POI 嵌入和权重张量
            # max_candidates = max(len(cands) for cands in candidate_pois) if candidate_pois else 1
            # candidate_poi_indices = torch.zeros(batch_size, max_candidates, dtype=torch.long, device=items.device)
            # candidate_poi_weights = torch.zeros(batch_size, max_candidates, device=items.device)  # 默认权重为 1
            # for i, cands in enumerate(candidate_pois):
            #     for j, poi_id in enumerate(cands):
            #         if not (0 <= poi_id <= self.item_num):
            #             print(f"无效 POI ID: {poi_id} 在 batch {i}, 最大有效 ID 为 {self.item_num}")
            #             poi_id = -1
            #         candidate_poi_indices[i, j] = poi_id
            #         assert len(poi_weights[i]) > j, f"POI 权重列表长度不足: {len(poi_weights[i])} < {j}"
            #         candidate_poi_weights[i, j] = poi_weights[i][j] if j < len(poi_weights[i]) else 0.0
            #     for j in range(len(cands), max_candidates):
            #         candidate_poi_indices[i, j] = self.PAD_IDX
            #         candidate_poi_weights[i, j] = 0.0

            # # 生成候选 POI 嵌入
            # candidate_poi_embeds = self.item_embeddings(candidate_poi_indices)  # [batch_size, max_candidates, emb_dim]

            # # 计算 POI 分数，应用权重
            # poi_scores = torch.matmul(poi_rep_diffu.unsqueeze(1), candidate_poi_embeds.transpose(-1, -2)).squeeze(1)
            # # 应用权重调整 POI 分数
            # poi_scores = poi_scores + 0.2 * (candidate_poi_weights)
            # mask = (candidate_poi_indices == self.PAD_IDX)
            # poi_scores = poi_scores.masked_fill(mask, -1e9)  # [batch_size, max_candidates]
            # _, top_k_pois = torch.topk(poi_scores, k=self.top_k_pois, dim=-1)
            # top_k_pois = torch.take_along_dim(candidate_poi_indices, top_k_pois, dim=1)

            return poi_rep_diffu, tile_rep_diffu


def create_model_diffu(args, quadkey_vocab_size, tile_vocab_size, tile_to_poi, poi_to_tile):
    diffu_pre = DiffuRec(args)
    return Att_Diffuse_model(diffu_pre, args, quadkey_vocab_size, tile_vocab_size, tile_to_poi, poi_to_tile)

