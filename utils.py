import torch.utils.data as data_utils
import torch
import torch.nn as nn
from datetime import datetime


class TrainDataset(data_utils.Dataset):
    def __init__(self, id2seq, max_len):
        self.id2seq = id2seq
        self.max_len = max_len  

    def __len__(self):
        return len(self.id2seq)

    def __getitem__(self, index):
        seq = self._getseq(index)
        labels = [seq[-1][0]]  # 标签是序列的最后一个元素(就是最后一个交互物品的序号),最后一个元素是元组，取元组的第一个
        last_time = seq[-1][1]
        tokens = seq[:-1] 
        tokens = [[item[0], int(item[1].timestamp())] for item in tokens]
        tokens = tokens[-self.max_len:]  # 保证 tokens 的长度不超过 max_len
        mask_len = self.max_len - len(tokens)  
        if mask_len > 0:
            mask_len = mask_len - 1  # 序列最长就是50，所以这里减了1。后面使用的序列长度都将是1
        else:
            tokens = tokens[1:]

        tokens = [[0, 0]] * mask_len + tokens + [[0, int(last_time.timestamp())]]  # 计算序列长度与 max_len 的差值    # 使用零填充序列的前面部分，使其长度等于 max_len
        # 最后一个元素是[0, 目标时间的时间戳]

        return torch.LongTensor(tokens), torch.LongTensor(labels)
        # longTensor期待转入的是一个格式统一的列表[1,666,7,...]，不允许其他值存在(比如[(1,0),(1,2),1]这种是不行的)

    def _getseq(self, idx):
        return self.id2seq[idx] 


class Data_Train():
    def __init__(self, data_train, args):
        self.u2seq = data_train
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        self.split_onebyone()  # 我搞不懂这样分割的作用是什么？后面的数据里也看不出用意？

    def split_onebyone(self):
        self.id_seq = {}
        self.id_seq_user = {}
        idx = 0
        for user_temp, seq_temp in self.u2seq.items():
            for star in range(len(seq_temp)-1):
                self.id_seq[idx] = seq_temp[:star+2]
                self.id_seq_user[idx] = user_temp
                idx += 1

    def get_pytorch_dataloaders(self):
        dataset = TrainDataset(self.id_seq, self.max_len)
        # 在dataset放入torch的dataloader函数之前，需要做好len()求数据集大小的函数，还有getitem()根据索引返回一条数据的函数
        # 最后一个物品的交互时间也是已知的、输入的量，输入的不仅仅是交互序列，该怎么组织代码？
        return data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)  
        # 这个dataset也只是一个对象。用getitem函数之后才返回了训练序列和标签
        # 在最开始的字典里，是序号：物品列表。那么加入：序号：[(物品1，时间1),(物品2，时间2),(物品3，时间3),...])。
        # 然后读取的时候，把时间单独读取进来？具体可能要查torch的dataloader函数了


class ValDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = [self.u2answer[user][0][0]]
        last_time = self.u2answer[user][0][1]
        seq = [[item[0], int(item[1].timestamp())] for item in seq]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        if padding_len > 0:
            padding_len = padding_len - 1
        else:
            seq = seq[1:]
        seq = [[0, 0]] * padding_len + seq + [[0, int(last_time.timestamp())]]

        return torch.LongTensor(seq),  torch.LongTensor(answer)


class Data_Val():
    def __init__(self, data_train, data_val, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2answer = data_val
        self.max_len = args.max_len

    def get_pytorch_dataloaders(self):
        dataset = ValDataset(self.u2seq, self.u2answer, self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader



class TestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2_seq_add, u2answer, max_len):
        self.u2seq = u2seq
        self.u2seq_add = u2_seq_add
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2seq_add[user]
        seq = [[item[0], int(item[1].timestamp())] for item in seq]
        # seq = self.u2seq[user]
        answer = [self.u2answer[user][0][0]]
        last_time = self.u2answer[user][0][1]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        if padding_len > 0:
            padding_len = padding_len - 1
        else:
            seq = seq[1:]
        seq = [[0, 0]] * padding_len + seq + [[0, int(last_time.timestamp())]]

        # print('attention！')
        # print(len(seq), answer)
        return torch.LongTensor(seq), torch.LongTensor(answer)


class Data_Test():
    def __init__(self, data_train, data_val, data_test, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2seq_add = data_val
        self.u2answer = data_test
        self.max_len = args.max_len

    def get_pytorch_dataloaders(self):
        dataset = TestDataset(self.u2seq, self.u2seq_add, self.u2answer, self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader


class CHLSDataset(data_utils.Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data_temp = self.data[index]
        seq = data_temp[:-1]
        seq = [[item[0], int(item[1].timestamp())] for item in seq]
        answer = [data_temp[-1][0]]
        last_time = data_temp[-1][1]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        if padding_len > 0:
            padding_len = padding_len - 1
        else:
            seq = seq[1:]
        seq = [[0, 0]] * padding_len + seq + [[0, int(last_time.timestamp())]]
        return torch.LongTensor(seq), torch.LongTensor(answer)


class Data_CHLS():
    def __init__(self, data, args):
        self.batch_size = args.batch_size
        self.max_len = args.max_len
        self.data = data

    def get_pytorch_dataloaders(self):
        dataset = CHLSDataset(self.data, self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader


def get_norm_time96(time):
    hour = time.hour
    minute = time.minute
    
    ans = minute//15 + 4*hour
    
    return ans

def get_day_norm7(time):
    # day_number = time.dayofweek
    day_number = time.weekday() 
    return day_number

