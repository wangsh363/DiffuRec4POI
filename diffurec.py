import torch.nn as nn
import torch as th
from step_sample import create_named_schedule_sampler
import numpy as np
import math
import torch
import torch.nn.functional as F
from utils import get_day_norm7, get_norm_time96
from datetime import datetime

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar in the limit of num_diffusion_timesteps. Beta schedules may be added, but should not be removed or changed once they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,)
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(num_diffusion_timesteps,lambda t: 1-np.sqrt(t + 0.0001),  )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(num_diffusion_timesteps, lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,)
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        if beta_end > 1:
            beta_end = scale * 0.001 + 0.01
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  #scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(beta_start, beta_mid, 10, dtype=np.float64)
        second_part = np.linspace(beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64)
        return np.concatenate([first_part, second_part])
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and produces the cumulative product of (1-beta) up to that part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):  ## 2000
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, but shifts towards left interval starting from 0
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)



def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    """
    时间编码函数：支持输入形状 [batch_size, seq_len]
    """
    tau = tau.float()
    if arg:
        v1 = f(torch.matmul(tau.unsqueeze(-1), w) + b, arg)  # 添加维度适配矩阵乘法
    else:
        v1 = f(torch.matmul(tau.unsqueeze(-1), w) + b)
    v2 = torch.matmul(tau.unsqueeze(-1), w0) + b0
    return torch.cat([v1, v2], -1)  # 在最后一维拼接


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))  # 线性部分参数
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))  # 非线性部分
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin  # 使用 sin 作为激活函数

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, none_args, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        """
        x: [batch_size, seq_len]，输入时间数据
        返回: [batch_size, seq_len, out_dim]，时间编码后的数据
        """

        # 检查参数变化
        # print('注意')
        # print(self.l1.w)
        # print(self.l1.b)

        return self.l1(x)  # 直接传入整个时间序列


def rotate(head, relation, hidden, device):
    pi = 3.14159265358979323846
        
    re_head, im_head = torch.chunk(head, 2, dim=1)  # 沿着第一维度拆分虚部和实部(?可能这个拆分的维度要变)

    embedding_range = nn.Parameter(
                    torch.Tensor([(24.0 + 2.0) / hidden]),   #  用来根据 hidden 值调整关系的范围（为什么是24+2?）
                    requires_grad=False
            ).to(device)


    phase_relation = relation/(embedding_range/pi)  # 将 relation 转换为相位

    # 计算旋转的实部和虚部
    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)

    # 应用旋转操作
    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation

    # 拼接实部和虚部
    score = torch.cat([re_score, im_score], dim = 1)
    return score


# 旋转的函数
def rotate_batch(head, relation, hidden, device):
    # head形状为(batch_size, seq_len, hidden_dim * 2),上下两层是复数
    # relation的形状是(batch_size, seq_len, hidden_dim)
    pi = 3.14159265358979323846

    re_head, im_head = torch.chunk(head, 2, dim=2)  # 拆分成实部和虚部

    # Make phases of relations uniformly distributed in [-pi, pi]
    embedding_range = nn.Parameter(
        torch.Tensor([(24.0 + 2.0) / hidden]),
        requires_grad=False
    ).to(device)

    phase_relation = relation / (embedding_range / pi)

    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)

    # 应用旋转操作
    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation

    score = torch.cat([re_score, im_score], dim=2)
    return score  # 形状为 (batch_size, seq_len, hidden_dim * 2)。


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


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


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size*4)
        self.w_2 = nn.Linear(hidden_size*4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat([1,1,1,corr.shape[-1]])
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class Transformer_rep(nn.Module):
    def __init__(self, args):
        super(Transformer_rep, self).__init__()
        self.hidden_size = args.hidden_size  # 隐藏层大小
        self.heads = 4  # 多头注意力机制的头数
        self.dropout = args.dropout  
        self.n_blocks = args.num_blocks  # Transformer 块的数量 
        # 一个包含多个 TransformerBlock 的模块列表，每个块都是一个独立的 Transformer 编码器。
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.heads, self.dropout) for _ in range(self.n_blocks)])

    def forward(self, hidden, mask):
        # hidden: 输入的特征表示，形状为 (batch_size, seq_len, hidden_size)。
        # mask: 序列掩码，形状为 (batch_size, seq_len)，用于标识哪些位置是有效的（非填充值）。
        for transformer in self.transformer_blocks:
            hidden = transformer.forward(hidden, mask)
        # 返回经过所有 Transformer 块编码后的 hidden。形状不变。
        return hidden


class Diffu_xstart(nn.Module):
    def __init__(self, hidden_size, args):
        super(Diffu_xstart, self).__init__()
        self.hidden_size = hidden_size

        # 线性层，用于对物品表示、加噪表示和时间步嵌入进行变换。
        self.linear_item = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_xt = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size)

        # time_embed: 时间步嵌入模块，将时间步 t 转换为嵌入表示。
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, time_embed_dim), SiLU(), nn.Linear(time_embed_dim, self.hidden_size))

        # fuse_linear: 融合线性层，用于将多个特征融合。
        self.fuse_linear = nn.Linear(self.hidden_size*3, self.hidden_size)

        # 新增的全连接层，用于把目标时间与diffu_rep的拼接输出成正式的目标向量输出
        # self.fc_out = nn.Linear(self.hidden_size * 2, self.hidden_size) 

        # 新增的全连接层，用于把物品的时间与物品的嵌入的拼接输出成物品的新嵌入
        # self.fc_item_out = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # att: 注意力机制模块（Transformer_rep）
        self.att = Transformer_rep(args)
        # self.mlp_model = nn.Linear(self.hidden_size, self.hidden_size)
        # self.gru_model = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        # self.gru_model = nn.GRU(self.hidden_size, self.hidden_size, num_layers=args.num_blocks, batch_first=True)

        # 时间编码层
        # 对于继承了nn.modlue的类，先初始化这个类，然后直接使用这个类的对象，就相当于调用了forward函数。
        # 这里的hidden_dim是指time2vec层的
        self.time2vec = Time2Vec('sin', 128, int(hidden_size ))  #  **!!!**  用旋转的方法的话，要除以2
        self.time2vec_day = Time2Vec('sin', 128, int(hidden_size ))  # 对时和周的编码要用不同的层

        # 不确定性系数、防止过拟合的dropout、层归一化系数
        self.lambda_uncertainty = args.lambda_uncertainty
        self.dropout = nn.Dropout(args.dropout)
        self.norm_diffu_rep = LayerNorm(self.hidden_size)

    # 将时间步t转化为嵌入表示
    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, rep_item, x_t, t, TimeStamp, mask_seq, item_tag):
        # rep_item是历史交互序列嵌入，x_t是加噪后的目标向量，t是时间步，mask_seq是序列掩码
        # (mask_seq不是位置编码，就只是一种掩码)
        emb_t = self.time_embed(self.timestep_embedding(t, self.hidden_size))  # 对时间步进行编码
        # print(x_t.size())  # torch.Size([512, 128])
        # print(x_t.unsqueeze(1).size())  # torch.Size([512, 1, 128])
        # x_t = x_t + emb_t 
                
        # 生成不确定性系数 lambda_uncertainty，即λ
        lambda_uncertainty = th.normal(mean=th.full(rep_item.shape, self.lambda_uncertainty), 
        std=th.full(rep_item.shape, self.lambda_uncertainty)).to(x_t.device)  ## distribution
        # 如果要创建系数的话。miu_uncertainty是时间的不确定性系数


        # ####  Attention：把整理好的向量(z1,z2,...zn)放入tranformer中

        # # 对时间戳进行编码
        # # 首先把时间戳按照日和周来划分。初始时间向量的大小是：(batch_size, max_len)，我需要拆成同样大小的日向量和周向量
        formatted_times = [[datetime.fromtimestamp(ts) for ts in seq] for seq in TimeStamp.tolist()]
        norm_times = [[get_norm_time96(time) / 96 for time in row] for row in formatted_times]  # 时间归一化到[0,1]
        day_times = [[get_day_norm7(time)/7 for time in row] for row in formatted_times]  # 星期归一化到[0,1]
        input_seq_time = torch.tensor(norm_times,dtype=torch.float).to(x_t.device)   # [512, 50]
        input_seq_day_time = torch.tensor(day_times,dtype=torch.float).to(x_t.device)

        # # 输入的大小为(batch_size, max_len)，输出的大小为(batch_size, max_len, hidden_size)
        time_emb_norm = self.time2vec(input_seq_time)  
        time_emb_day = self.time2vec_day(input_seq_day_time)
        time_target =(0.7 * time_emb_norm + 0.3 * time_emb_day)[:, -1, :]


        # 原始代码
        # x_t = x_t + emb_t
        # rep_diffu = self.att(rep_item + lambda_uncertainty * x_t.unsqueeze(1), mask_seq)  #  rep_diffu的大小是(512,50,128)

        # 将时间向量和正常的向量进行合并
        # rep_item[:, -1, :] = x_t   # 把最后一个空向量换成x_s
        x_t = x_t + emb_t
        time_emb_all = 0.7 * time_emb_norm + 0.3 * time_emb_day  # 大小是[512, 50, 128]，rep_diffu也是[512, 50, 128]
        # 直接相加
        rep_diffu = self.att(rep_item + lambda_uncertainty * x_t.unsqueeze(1) + time_emb_all , mask_seq)
        # 和物品嵌入拼接之后经过全连接层
        # rep_item_add_time = torch.cat((rep_item, time_emb_all), dim=2)
        # rep_item_afteradd = self.fc_item_out(rep_item_add_time)
        # rep_diffu = self.att(rep_item_afteradd + lambda_uncertainty * x_t.unsqueeze(1), mask_seq)


        # # 使用旋转操作融合向量
        # x_t = x_t + emb_t
        # Rotate_tmp_norm = rotate_batch(rep_item + lambda_uncertainty * x_t.unsqueeze(1), 
        # time_emb_norm, int(self.hidden_size / 2), x_t.device)
        # Rotate_tmp_day = rotate_batch(rep_item + lambda_uncertainty * x_t.unsqueeze(1), 
        # time_emb_day, int(self.hidden_size / 2), x_t.device)
        # # 对日和周旋转融合后的向量加权
        # Rotate_tmp = 0.7 * Rotate_tmp_norm + 0.3 * Rotate_tmp_day
        # rep_diffu = self.att(Rotate_tmp, mask_seq)

        rep_diffu = self.norm_diffu_rep(self.dropout(rep_diffu))

        # out就是重建后的x0，rep_diffu是(h1,h2,...,hn)
        # out = rep_diffu[:, -1, :]
        out = rep_diffu[:, -2, :]
        
        # 用重建好的x0加上目标时间
        out = out + time_target  # size是[512, 128])

        # 用旋转的方式
        # out = rotate(out, time_target, int(self.hidden_size / 2), x_t.device)

        # 用拼接的方式(沿第二维)
        # out_add_time = torch.cat((out, time_target), dim=1)
        # out = self.fc_out(out_add_time)


        
        ###############################################################################################################
        
        #### GRU
        # output, hn = self.gru_model(rep_item + lambda_uncertainty * x_t.unsqueeze(1))
        # output = self.norm_diffu_rep(self.dropout(output))
        # out = output[:,-1,:]
        ## # out = hn.squeeze(0)
        # rep_diffu = None
        ####
        
        ### MLP
        # output = self.mlp_model(rep_item + lambda_uncertainty * x_t.unsqueeze(1))
        # output = self.norm_diffu_rep(self.dropout(output))
        # out = output[:,-1,:]
        # rep_diffu = None
        ###
        
        # out = out + self.lambda_uncertainty * x_t
        # time_target = None
        
        return out, rep_diffu, item_tag, time_target


class DiffuRec(nn.Module):
    def __init__(self, args,):  # 初始化扩散模型的参数
        super(DiffuRec, self).__init__()
        self.hidden_size = args.hidden_size
        self.schedule_sampler_name = args.schedule_sampler_name
        self.diffusion_steps = args.diffusion_steps
        self.use_timesteps = space_timesteps(self.diffusion_steps, [self.diffusion_steps])

        # 计算扩散过程中的 betas 和 alphas（这些控制噪声添加的程度）
        self.noise_schedule = args.noise_schedule
        betas = self.get_betas(self.noise_schedule, self.diffusion_steps)
         # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas

        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)  # 表示累积乘积，用于扩散过程中的标准化。
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])  # 前一个时间步的累积乘积，帮助计算后续时间步的变化。

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        self.num_timesteps = int(self.betas.shape[0])
       
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.num_timesteps)  ## lossaware (schedule_sample)
        self.timestep_map = self.time_map()
        self.rescale_timesteps = args.rescale_timesteps
        self.original_num_steps = len(betas)

        self.xstart_model = Diffu_xstart(self.hidden_size, args)

    def get_betas(self, noise_schedule, diffusion_steps):
        betas = get_named_beta_schedule(noise_schedule, diffusion_steps)  ## array, generate beta
        return betas
    

    def q_sample(self, x_start, t, noise=None, mask=None):  # q_sample：根据给定的初始数据 x_start 和当前时间步 t，生成在该时间步 t 时的噪声数据 x_t。
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch. 
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)

        assert noise.shape == x_start.shape
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise  ## reparameter trick
        )  ## genetrate x_t based on x_0 (x_start) with reparameter trick

        if mask == None:
            return x_t
        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)  ## mask: [0,0,0,1,1,1,1,1]
            return th.where(mask==0, x_start, x_t)  ## replace the output_target_seq embedding (x_0) as x_t

    def time_map(self):
        timestep_map = []
        for i in range(len(self.alphas_cumprod)):
            if i in self.use_timesteps:
                timestep_map.append(i)
        return timestep_map

    # def scale_t(self, ts):
    #     map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
    #     new_ts = map_tensor[ts]
    #     # print(new_ts)
    #     if self.rescale_timesteps:
    #         new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
    #     return new_ts

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior: 
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  ## \mu_t
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, TimeStamp, mask_seq): 
        # 有一个诡异的报错？这里加一个无用的参数
        item_tag = None
        # 计算在给定当前时间步 t 的带噪声输入 x_t 的情况下，下一步（即时间步 t-1）的均值和对数方差。
        model_output, rep_diffu, item_tag, time_target = self.xstart_model(rep_item, x_t, self._scale_timesteps(t), TimeStamp, mask_seq, item_tag)
        
        x_0 = model_output  ##output predict
        # x_0 = self._predict_xstart_from_eps(x_t, t, model_output)  ## eps predict
        
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)
        
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)  ## x_start: candidante item embedding, x_t: inputseq_embedding + outseq_noise, output x_(t-1) distribution
        return model_mean, model_log_variance, time_target

    def p_sample(self, item_rep, noise_x_t, t, TimeStamp, mask_seq):  
        # 给定当前时间步的噪声数据 x_t，生成去噪后的数据 x_(t-1)。通过采样，逐步将噪声数据恢复到原始数据。
        model_mean, model_log_variance, time_target = self.p_mean_variance(item_rep, noise_x_t, t, TimeStamp, mask_seq)
        noise = th.randn_like(noise_x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(noise_x_t.shape) - 1))))  # no noise when t == 0
        sample_xt = model_mean + nonzero_mask * th.exp(0.5 * model_log_variance) * noise  ## sample x_{t-1} from the \mu(x_{t-1}) distribution based on the reparameter trick
        return sample_xt, time_target

    def reverse_p_sample(self, item_rep, noise_x_t, TimeStamp, mask_seq):  # 通过迭代从时间步 T 到 0，逐步去噪，最终得到没有噪声的原始数据。
        device = next(self.xstart_model.parameters()).device
        indices = list(range(self.num_timesteps))[::-1]
        
        for i in indices: # from T to 0, reversion iteration  
            t = th.tensor([i] * item_rep.shape[0], device=device)
            with th.no_grad():
                noise_x_t, time_target = self.p_sample(item_rep, noise_x_t, t, TimeStamp, mask_seq)
        return noise_x_t, time_target 

    def forward(self, item_rep, item_tag, TimeStamp, mask_seq):
        noise = th.randn_like(item_tag)  # 和初始物品嵌入形状一致的随机噪声
        # 使用 schedule_sampler 采样时间步 t 和对应的权重 weights。
        t, weights = self.schedule_sampler.sample(item_rep.shape[0], item_tag.device) ## t is sampled from schedule_sampler
        
        # t = self.scale_t(t)
        x_t = self.q_sample(item_tag, t, noise=noise)  # 调用 q_sample 方法，对目标表示 item_tag 进行正向扩散（加噪）。
        
        # eps, item_rep_out = self.xstart_model(item_rep, x_t, self._scale_timesteps(t), mask_seq)  ## eps predict
        # x_0 = self._predict_xstart_from_eps(x_t, t, eps)

        # 调用 xstart_model，预测目标表示 x_0 和扩散后的物品表示 item_rep_out
        x_0, item_rep_out, item_tag, time_target = self.xstart_model(item_rep, x_t, self._scale_timesteps(t), TimeStamp, mask_seq, item_tag)  ##output predict

        # xstart_model 是一个神经网络模块，负责从扩散后的表示 x_t 中恢复目标表示 x_0。
        # item_rep 是历史交互序列（不包括目标序列）

        return x_0, item_rep_out, weights, t, time_target   # 返回预测结果x0、(h1,h2,...,hn)、权重和时间步。


