import pickle
import pandas as pd
from datetime import datetime

# 加载 CSV 文件
path = 'TKY_train.csv'  # 替换为实际路径
df = pd.read_csv(path, usecols=['trajectory_id', 'POI_id', 'local_time', 'latitude', 'longitude'])

# 创建用户-物品交互字典并收集 raw_poi_id
user_item_dict = {}
all_raw_poi_ids = set()
for _, row in df.iterrows():
    user_id = int(row['trajectory_id'])
    item_id = int(row['POI_id'])
    timestamp = row['local_time']
    latitude = float(row['latitude'])
    longitude = float(row['longitude'])
    entry = (item_id, timestamp, latitude, longitude)
    if user_id not in user_item_dict:
        user_item_dict[user_id] = [entry]
    else:
        user_item_dict[user_id].append(entry)
    all_raw_poi_ids.add(item_id)

# 过滤序列长度小于 4 的用户
user_item_dict = {user_id: value for user_id, value in user_item_dict.items() if len(value) >= 4}

# 更新 raw_poi_id 集合，仅包含过滤后数据中的 POI
filtered_raw_poi_ids = set()
for seq in user_item_dict.values():
    for item_id, _, _, _ in seq:
        filtered_raw_poi_ids.add(item_id)
print(f"过滤后 raw_poi_id 数量: {len(filtered_raw_poi_ids)}")

# 构建 smap：键为连续 ID，值为 raw_poi_id
smap = {idx : raw_poi_id for idx, raw_poi_id in enumerate(sorted(filtered_raw_poi_ids))}
print(f"smap 大小: {len(smap)}")
print(f"smap 键范围: [{min(smap.keys())}, {max(smap.keys())}], 值示例: {list(smap.values())[:5]}")

# 更新 item_entities.dict
with open('item_entities.dict', 'w') as file:
    for mapped_id, raw_poi_id in smap.items():
        file.write(f"{mapped_id} {raw_poi_id}\n")
print("item_entities.dict 已更新")

# 构建 umap
umap = {i: i-1 for i in range(1, 10377)}

# 将用户 ID 重新映射为连续的自然数
sorted_keys = sorted(user_item_dict.keys())
new_dict = {new_key: user_item_dict[old_key] for new_key, old_key in enumerate(sorted_keys)}

# 初始化 train、val、test
test = {}
val = {}
train = {}
for key, value in new_dict.items():
    test[key] = [value[-1]]
    val[key] = [value[-2]]
    train[key] = value[:-2]

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