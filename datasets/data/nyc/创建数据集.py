import pickle
import pandas as pd
from datetime import datetime

# 加载原始的 dataset.pkl 文件
with open('dataset.pkl', 'rb') as f:
    data_dict = pickle.load(f)
print('字典最外层的键：', list(data_dict.keys()))

# 读取 item_entities.dict 文件，构建 smap
with open('item_entities.dict', 'r') as file:
    smap = {int(key): int(value) for line in file for key, value in [line.split()]}

# 构建 umap，假设用户 ID 从 1 到 10376
umap = {i: i-1 for i in range(1, 10377)}

# 读取 CSV 文件，包含经纬度
path = 'NYC_train.csv'  # 请替换为你的实际文件路径
df = pd.read_csv(path, usecols=['trajectory_id', 'POI_id', 'local_time', 'latitude', 'longitude'])

# 创建一个空字典，存储用户-物品交互序列（包含时间和经纬度）
user_item_dict = {}

# 遍历 CSV 的每一行，将数据整理到字典中
for _, row in df.iterrows():
    user_id = int(row['trajectory_id'])  # 用户 ID
    item_id = int(row['POI_id'])         # 物品 ID
    timestamp = row['local_time']        # 时间戳（字符串格式）
    latitude = float(row['latitude'])    # 纬度
    longitude = float(row['longitude'])  # 经度

    # 构造包含物品 ID、时间戳、经纬度的元组
    entry = (item_id, timestamp, latitude, longitude)

    # 如果用户 ID 不在字典中，初始化列表
    if user_id not in user_item_dict:
        user_item_dict[user_id] = [entry]
    else:
        user_item_dict[user_id].append(entry)

# 过滤掉序列长度小于 4 的用户
user_item_dict = {user_id: value for user_id, value in user_item_dict.items() if len(value) >= 4}

# 将用户 ID 重新映射为连续的自然数
sorted_keys = sorted(user_item_dict.keys())
new_dict = {new_key: user_item_dict[old_key] for new_key, old_key in enumerate(sorted_keys)}

# 初始化 train、val、test 字典
test = {}
val = {}
train = {}

# 拆分数据为 train、val、test
for key, value in new_dict.items():
    test[key] = [value[-1]]   # 最后一个交互作为测试集
    val[key] = [value[-2]]    # 倒数第二个交互作为验证集
    train[key] = value[:-2]   # 其余的作为训练集

# 打印训练集的键和部分数据，确认格式
print("train keys:", train.keys())
print("train sample:", train[0])  # 打印第一个用户的训练数据

# 构造结果字典
result_dict = {
    'train': train,
    'val': val,
    'test': test,
    'umap': umap,
    'smap': smap
}

# 保存到新的 pkl 文件
with open('dataset.pkl', 'wb') as f:
    pickle.dump(result_dict, f)

print("数据已保存到 dataset.pkl")

############
# 字典最外层的键： ['train', 'val', 'test', 'umap', 'smap']
# train keys: dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
# train sample: [(244, '2010-06-06 22:11:04', 37.7826046833, -122.4076080167),
# 数据已保存到 dataset.pkl
############