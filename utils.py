import torch.utils.data as data_utils
import torch
import torch.nn as nn
from datetime import datetime
from pyquadkey2 import quadkey as pqk
from torchtext.vocab import build_vocab_from_iterator
from nltk import ngrams
import pickle
import os

def save_vocab_cache(cache_dir, dataset_name, quadkey_vocab, tile_vocab, tiles, tile_to_poi, poi_to_tile):
    """保存词汇表和映射到 pickle 文件"""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_path = os.path.join(cache_dir, f"{dataset_name}_vocab_cache.pkl")
    cache_data = {
        'quadkey_vocab': quadkey_vocab,
        'tile_vocab': tile_vocab,
        'tiles': tiles,
        'tile_to_poi': tile_to_poi,
        'poi_to_tile': poi_to_tile
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"词汇表缓存已保存至: {cache_path}")


def load_vocab_cache(cache_dir, dataset_name):
    """从 pickle 文件加载词汇表和映射"""
    cache_path = os.path.join(cache_dir, f"{dataset_name}_vocab_cache.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        print(f"从缓存加载词汇表: {cache_path}")
        return (cache_data['quadkey_vocab'], cache_data['tile_vocab'],
                cache_data['tiles'], cache_data['tile_to_poi'], cache_data['poi_to_tile'])
    else:
        print(f"缓存文件不存在: {cache_path}")
        return None

def calculate_boundary(data_dict):
    lons = []
    lats = []
    for split in ['train', 'val', 'test']:
        if split not in data_dict:
            continue
        for seq in data_dict[split].values():
            for _, _, _, lat, lon in seq:
                if lat != 0.0 or lon != 0.0:
                    lons.append(lon)
                    lats.append(lat)
    if not lons or not lats:
        print("警告: 无有效经纬度数据")
        return [-180, -90, 180, 90]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    buffer = 0.1  # 增大缓冲区
    min_lon, max_lon = max(min_lon - buffer, -180), min(max_lon + buffer, 180)
    min_lat, max_lat = max(min_lat - buffer, -90), min(max_lat + buffer, 90)
    print(f"POI经度范围: [{min(lons)}, {max(lons)}]")
    print(f"POI纬度范围: [{min(lats)}, {max(lats)}]")
    print(f"计算的边界: [{min_lon}, {min_lat}, {max_lon}, {max_lat}]")
    return [min_lon, min_lat, max_lon, max_lat]


class Quadkey:
    def __init__(self, qk_str):
        self.qk_str = qk_str

    @staticmethod
    def from_geo(coords, level):
        qk = pqk.from_geo(coords, level)
        return Quadkey(qk)

    def __str__(self):
        return str(self.qk_str)


def build_quadkey_vocab(data_dict, lod=17):
    """为所有数据构建统一的 Quadkey 词汇表"""
    all_quadkeys = []
    for split in ['train', 'val', 'test']:
        for seq in data_dict[split].values():
            for _, _, _, lat, lon in seq:
                if lat != 0.0 or lon != 0.0:
                    qk = Quadkey.from_geo((lat, lon), lod)
                    qk_str = str(qk)
                    qk_ngrams = ' '.join([''.join(x) for x in ngrams(qk_str, 6)]) if len(qk_str) >= 6 else qk_str
                    all_quadkeys.append(qk_ngrams.split())

    def token_iterator():
        for tokens in all_quadkeys:
            yield tokens

    quadkey_vocab = build_vocab_from_iterator(token_iterator(), specials=['<unk>', '<pad>'])
    print(f"QUADKEY_VOCAB type: {type(quadkey_vocab)}")
    print(f"Quadkey 词汇表大小: {len(quadkey_vocab)}")
    return quadkey_vocab


class QuadTree:
    def __init__(self, boundary, max_depth=5, max_items=10):
        self.boundary = boundary  # [min_lon, min_lat, max_lon, max_lat]
        self.max_depth = max_depth
        self.max_items = max_items
        self.items = []  # 存储POI: (lon, lat, poi_id)
        self.children = None  # 子节点
        self.id = None  # 瓦片ID
        self.depth = 0
        self.poi_ids = set()

    def insert(self, item):
        if not self.contains(item[0], item[1]):
            return False
        if self.children is None:
            # 仅在叶节点添加POI
            self.items.append(item)
            self.poi_ids.add(item[2])
        if self.children is None and len(self.items) > self.max_items and self.depth < self.max_depth:
            self.split()
        if self.children is not None:
            for child in self.children:
                child.insert(item)
        return True

    def split(self):
        mid_lon = (self.boundary[0] + self.boundary[2]) / 2
        mid_lat = (self.boundary[1] + self.boundary[3]) / 2
        self.children = [
            QuadTree([self.boundary[0], mid_lat, mid_lon, self.boundary[3]], self.max_depth, self.max_items),
            QuadTree([mid_lon, mid_lat, self.boundary[2], self.boundary[3]], self.max_depth, self.max_items),
            QuadTree([self.boundary[0], self.boundary[1], mid_lon, mid_lat], self.max_depth, self.max_items),
            QuadTree([mid_lon, self.boundary[1], self.boundary[2], mid_lat], self.max_depth, self.max_items)
        ]
        for child in self.children:
            child.depth = self.depth + 1
        inserted_pois = set()
        for item in self.items:
            inserted = False
            for child in self.children:
                if child.insert(item):
                    inserted_pois.add(item[2])
                    inserted = True
                    break
            if not inserted:
                print(f"POI {item[2]} 未插入到任何子节点，坐标: ({item[0]}, {item[1]})")
        if len(inserted_pois) != len(self.poi_ids):
            print(f"分裂时丢失POI: 原始 {len(self.poi_ids)} 个，插入 {len(inserted_pois)} 个")
        self.items = []
        self.poi_ids = set()

    def contains(self, lon, lat):
        return (self.boundary[0] <= lon <= self.boundary[2] and
                self.boundary[1] <= lat <= self.boundary[3])

    def get_leaves(self, tiles=None, tile_id=0):
        if tiles is None:
            tiles = []
        if self.children is None:
            self.id = tile_id
            tiles.append(self)
            # print(f"Tile {tile_id}: Boundary {self.boundary}, POI IDs {self.poi_ids}")
            return tiles, tile_id + 1
        for child in self.children:
            tiles, tile_id = child.get_leaves(tiles, tile_id)
        return tiles, tile_id


def generate_tiles(data_dict, boundary, max_depth=5, max_items=10):
    qt = QuadTree(boundary, max_depth, max_items)
    smap_reverse = data_dict.get('smap_reverse', {})
    unmapped_pois = []
    valid_pois = []
    for split in ['train', 'val', 'test']:
        for seq in data_dict[split].values():
            for raw_poi_id, _, _, lat, lon in seq:
                if lat != 0.0 or lon != 0.0:
                    mapped_poi_id = smap_reverse.get(raw_poi_id, -1)
                    if mapped_poi_id == -1:
                        unmapped_pois.append((raw_poi_id, lat, lon, "无效映射"))
                        continue
                    if not qt.insert((lon, lat, mapped_poi_id)):
                        unmapped_pois.append((raw_poi_id, lat, lon, "超出边界"))
                    else:
                        valid_pois.append((lon, lat, mapped_poi_id))
                else:
                    unmapped_pois.append((raw_poi_id, lat, lon, "无效经纬度"))
    print(f"有效插入的POI数量: {len(valid_pois)}")
    print(f"未插入的POI数量: {len(unmapped_pois)}")
    if unmapped_pois:
        print("未插入POI示例:")
        for poi in unmapped_pois[:10]:
            print(f"POI: {poi[0]}, 经纬度: ({poi[1]}, {poi[2]}), 原因: {poi[3]}")
    tiles, _ = qt.get_leaves()
    return tiles


def map_to_tile(tiles, lon, lat):
    for tile in tiles:
        if tile.contains(lon, lat):
            return tile.id
    return -1


def build_tile_vocab(data_dict, max_depth=10, max_items=50):
    boundary = calculate_boundary(data_dict)
    tiles = generate_tiles(data_dict, boundary, max_depth, max_items)
    for i, tile in enumerate(tiles, 1):
        tile.id = i
    tile_vocab = {tile.id: tile for tile in tiles}
    tile_vocab[0] = None
    unk_tile_id = len(tiles) + 1
    tile_vocab[unk_tile_id] = None
    tile_to_poi = {tile.id: tile.poi_ids for tile in tiles}
    tile_to_poi[0] = set()
    tile_to_poi[unk_tile_id] = set()

    poi_to_tile = {}
    for tile_id, poi_ids in tile_to_poi.items():
        for poi_id in poi_ids:
            poi_to_tile[poi_id] = tile_id

    smap_reverse = data_dict.get('smap_reverse', {})
    all_poi_ids = set(smap_reverse.values())
    used_poi_ids = set()
    for split in ['train', 'val', 'test']:
        for seq in data_dict[split].values():
            for raw_poi_id, _, _, _, _ in seq:
                mapped_poi_id = smap_reverse.get(raw_poi_id, -1)
                if mapped_poi_id != -1:
                    used_poi_ids.add(mapped_poi_id)
    print(f"总POI ID数量: {len(all_poi_ids)}")
    print(f"实际使用的POI ID数量: {len(used_poi_ids)}")
    unused_poi_ids = all_poi_ids - used_poi_ids
    if unused_poi_ids:
        print(f"未使用的POI ID数量: {len(unused_poi_ids)}, 示例: {list(unused_poi_ids)[:10]}")

    mapped_poi_ids = set(poi_to_tile.keys())
    missing_pois = all_poi_ids - mapped_poi_ids
    if missing_pois:
        print(f"警告: {len(missing_pois)} 个 POI ID 未映射到任何瓦片: {list(missing_pois)[:10]}")
        if missing_pois.issubset(unused_poi_ids):
            print("所有未映射的POI ID均为未使用的数据")
        for poi_id in missing_pois:
            poi_to_tile[poi_id] = unk_tile_id
            tile_to_poi[unk_tile_id].add(poi_id)

    return tile_vocab, tiles, tile_to_poi, poi_to_tile


def build_data_vocabs(data_dict, cache_dir='./cache', dataset_name='gowalla'):
    """构建数据词汇表，并支持缓存"""
    cached_data = load_vocab_cache(cache_dir, dataset_name)
    if cached_data is not None:
        return cached_data

    # 如果没有缓存，重新计算
    quadkey_vocab = build_quadkey_vocab(data_dict)
    tile_vocab, tiles, tile_to_poi, poi_to_tile = build_tile_vocab(data_dict)

    # 保存到缓存
    save_vocab_cache(cache_dir, dataset_name, quadkey_vocab, tile_vocab, tiles, tile_to_poi, poi_to_tile)

    return quadkey_vocab, tile_vocab, tiles, tile_to_poi, poi_to_tile


class TrainDataset(data_utils.Dataset):
    def __init__(self, id2seq, max_len, quadkey_vocab, tiles, tile_vocab_size, tile_to_poi, poi_to_tile, lod=17, smap_reverse=None, cache_dir='./cache', dataset_name='gowalla'):
        self.id2seq = id2seq
        self.max_len = max_len
        self.quadkey_vocab = quadkey_vocab
        self.tiles = tiles
        self.tile_vocab_size = tile_vocab_size
        self.tile_to_poi = tile_to_poi
        self.poi_to_tile = poi_to_tile
        self.lod = lod
        self.smap_reverse = smap_reverse or {}
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name
        # 预计算 quadkeys 和 tile_ids
        # self.precomputed_data = self._precompute_quadkeys_and_tiles()
        # 预加载缓存功能
        # self.precomputed_data = self._load_precomputed_cache()
        # if self.precomputed_data is None:
        #     self.precomputed_data = self._precompute_quadkeys_and_tiles()
        #     self._save_precomputed_cache()

    # 如果保存预加载的话可以减少第一次整体计算的时间，但是对于lod等参数变化时不方便修改，记录在缓存中了
    # def _save_precomputed_cache(self):
    #     """保存预计算的 quadkeys 和 tile_ids 到 pickle 文件"""
    #     cache_path = os.path.join(self.cache_dir, f"{self.dataset_name}_train_precomputed.pkl")
    #     with open(cache_path, 'wb') as f:
    #         pickle.dump(self.precomputed_data, f)
    #     print(f"预计算数据已保存至: {cache_path}")

    # def _load_precomputed_cache(self):
    #     """从 pickle 文件加载预计算的 quadkeys 和 tile_ids"""
    #     cache_path = os.path.join(self.cache_dir, f"{self.dataset_name}_train_precomputed.pkl")
    #     if os.path.exists(cache_path):
    #         with open(cache_path, 'rb') as f:
    #             precomputed_data = pickle.load(f)
    #         print(f"从缓存加载预计算数据: {cache_path}")
    #         return precomputed_data
    #     return None

    def _precompute_quadkeys_and_tiles(self):
        """预计算所有序列的 quadkeys 和 tile_ids"""
        precomputed = {}
        for idx in range(len(self.id2seq)):
            seq = self._getseq(idx)
            tokens = seq[:-1]
            tokens = [[self.smap_reverse.get(item[0], -1), int(item[1].timestamp()), item[2], item[3], item[4]] for
                      item in tokens]
            tokens = tokens[-self.max_len:]
            mask_len = self.max_len - len(tokens)
            if mask_len > 0:
                mask_len = mask_len - 1
            else:
                tokens = tokens[1:]
            tokens = [[0, 0, 0, 0.0, 0.0]] * mask_len + tokens + [[0, int(seq[-1][1].timestamp()), 0, 0.0, 0.0]]

            quadkeys = []
            tile_ids = []
            coords = []
            for item in tokens:
                lat, lon = item[3], item[4]
                if lat == 0.0 and lon == 0.0:
                    quadkeys.append(['0'])
                    tile_ids.append(0)
                    coords.append([0.0, 0.0])
                else:
                    qk = Quadkey.from_geo((lat, lon), self.lod)
                    qk_str = str(qk)
                    qk_ngrams = ' '.join([''.join(x) for x in ngrams(qk_str, 6)]) if len(qk_str) >= 6 else qk_str
                    quadkeys.append(qk_ngrams.split())
                    tile_id = map_to_tile(self.tiles, lon, lat)
                    tile_id = tile_id if tile_id != -1 and tile_id < self.tile_vocab_size else self.tile_vocab_size - 1
                    tile_ids.append(tile_id)
                    coords.append([lat, lon])
            precomputed[idx] = (quadkeys, tile_ids, coords)
        return precomputed

    def __len__(self):
        return len(self.id2seq)

    def __getitem__(self, index):
        seq = self._getseq(index)
        raw_label = seq[-1][0]
        label = self.smap_reverse.get(raw_label, -1)
        labels = [label]
        unk_tile_id = self.tile_vocab_size - 1
        tile_label = self.poi_to_tile.get(label, unk_tile_id)
        tile_labels = [tile_label]
        last_time = seq[-1][1]
        last_uid = seq[-1][2]
        tokens = seq[:-1]
        tokens = [[self.smap_reverse.get(item[0], -1), int(item[1].timestamp()), item[2], item[3], item[4]] for item in tokens]
        tokens = tokens[-self.max_len:]
        mask_len = self.max_len - len(tokens)
        if mask_len > 0:
            mask_len = mask_len - 1
        else:
            tokens = tokens[1:]
        tokens = [[0, 0, 0, 0.0, 0.0]] * mask_len + tokens + [[0, int(last_time.timestamp()), last_uid, 0.0, 0.0]]
        items = [x[0] for x in tokens]
        timestamps = [x[1] for x in tokens]
        uids = [x[2] for x in tokens]
        coords = [[x[3], x[4]] for x in tokens]
        # quadkeys, tile_ids, coords = self.precomputed_data[index]
        # unk_index = self.quadkey_vocab['<unk>']
        # quadkey_indices = [[self.quadkey_vocab[token] if token in self.quadkey_vocab else unk_index for token in qk] for qk in quadkeys]
        # max_ngram_len = max(len(indices) for indices in quadkey_indices)
        # pad_index = self.quadkey_vocab['<pad>']
        # quadkey_indices = [indices + [pad_index] * (max_ngram_len - len(indices)) for indices in quadkey_indices]
        tile_ids = [self.poi_to_tile.get(key) for key in items]
        quadkey_indices = []
        return (torch.LongTensor(items),
                torch.LongTensor(timestamps),
                torch.LongTensor(uids),
                torch.LongTensor(quadkey_indices),
                torch.LongTensor(tile_ids),
                torch.LongTensor(labels),
                torch.LongTensor(tile_labels),
                torch.FloatTensor(coords))

    def _getseq(self, idx):
        return self.id2seq[idx]


class Data_Train:
    def __init__(self, data_train, args, quadkey_vocab, tiles, tile_vocab_size, tile_to_poi, poi_to_tile, smap_reverse):
        self.u2seq = data_train
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        self.quadkey_vocab = quadkey_vocab
        self.tiles = tiles
        self.tile_vocab_size = tile_vocab_size
        self.tile_to_poi = tile_to_poi
        self.poi_to_tile = poi_to_tile
        self.smap_reverse = smap_reverse
        self.split_onebyone()
        self.dataset_name = args.dataset
        self.cache_dir = './cache'

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
        dataset = TrainDataset(self.id_seq, self.max_len, self.quadkey_vocab, self.tiles, self.tile_vocab_size, self.tile_to_poi, self.poi_to_tile, smap_reverse=self.smap_reverse, cache_dir=self.cache_dir, dataset_name=self.dataset_name)
        return data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        items, timestamps, uids, quadkey_indices, tile_ids, labels, tile_labels, coords = zip(*batch)
        return (torch.stack(items),
                torch.stack(timestamps),
                torch.stack(uids),
                torch.stack(quadkey_indices),
                torch.stack(tile_ids),
                torch.stack(labels),
                torch.stack(tile_labels),
                torch.stack(coords))

class ValDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, quadkey_vocab, tiles, tile_vocab_size, tile_to_poi, poi_to_tile, lod=17, smap_reverse=None, cache_dir='./cache', dataset_name='gowalla'):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.quadkey_vocab = quadkey_vocab
        self.tiles = tiles
        self.tile_vocab_size = tile_vocab_size
        self.tile_to_poi = tile_to_poi
        self.poi_to_tile = poi_to_tile
        self.lod = lod
        self.smap_reverse = smap_reverse or {}
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name
        # self.precomputed_data = self._precompute_quadkeys_and_tiles()

    def _precompute_quadkeys_and_tiles(self):
        """预计算所有序列的 quadkeys 和 tile_ids"""
        precomputed = {}
        for idx, user in enumerate(self.users):
            seq = self.u2seq[user]
            seq = [[self.smap_reverse.get(item[0], -1), int(item[1].timestamp()), item[2], item[3], item[4]] for item in
                   seq]
            seq = seq[-self.max_len:]
            padding_len = self.max_len - len(seq)
            if padding_len > 0:
                padding_len = padding_len - 1
            else:
                seq = seq[1:]
            seq = [[0, 0, 0, 0.0, 0.0]] * padding_len + seq + [
                [0, int(self.u2answer[user][0][1].timestamp()), 0, 0.0, 0.0]]

            quadkeys = []
            tile_ids = []
            coords = []
            for item in seq:
                lat, lon = item[3], item[4]
                if lat == 0.0 and lon == 0.0:
                    quadkeys.append(['0'])
                    tile_ids.append(0)
                    coords.append([0.0, 0.0])
                else:
                    qk = Quadkey.from_geo((lat, lon), self.lod)
                    qk_str = str(qk)
                    qk_ngrams = ' '.join([''.join(x) for x in ngrams(qk_str, 6)]) if len(qk_str) >= 6 else qk_str
                    quadkeys.append(qk_ngrams.split())
                    tile_id = map_to_tile(self.tiles, lon, lat)
                    tile_id = tile_id if tile_id != -1 and tile_id < self.tile_vocab_size else self.tile_vocab_size - 1
                    tile_ids.append(tile_id)
                    coords.append([lat, lon])
            precomputed[idx] = (quadkeys, tile_ids, coords)
        return precomputed

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        raw_answer = self.u2answer[user][0][0]
        answer = self.smap_reverse.get(raw_answer, -1)
        answer = [answer]
        unk_tile_id = self.tile_vocab_size - 1
        tile_label = self.poi_to_tile.get(answer[0], unk_tile_id)
        tile_labels = [tile_label]
        last_time = self.u2answer[user][0][1]
        last_uid = self.u2answer[user][0][2]
        seq = [[self.smap_reverse.get(item[0], -1), int(item[1].timestamp()), item[2], item[3], item[4]] for item in seq]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        if padding_len > 0:
            padding_len = padding_len - 1
        else:
            seq = seq[1:]
        seq = [[0, 0, 0, 0.0, 0.0]] * padding_len + seq + [[0, int(last_time.timestamp()), last_uid, 0.0, 0.0]]
        items = [x[0] for x in seq]
        timestamps = [x[1] for x in seq]
        uids = [x[2] for x in seq]
        # quadkeys, tile_ids, coords = self.precomputed_data[index]
        coords = [[x[3], x[4]] for x in seq]
        tile_ids = [self.poi_to_tile.get(key) for key in items]
        # unk_index = self.quadkey_vocab['<unk>']
        # quadkey_indices = [[self.quadkey_vocab[token] if token in self.quadkey_vocab else unk_index for token in qk] for qk in quadkeys]
        # max_ngram_len = max(len(indices) for indices in quadkey_indices)
        # pad_index = self.quadkey_vocab['<pad>']
        # quadkey_indices = [indices + [pad_index] * (max_ngram_len - len(indices)) for indices in quadkey_indices]
        quadkey_indices = []
        return (torch.LongTensor(items),
                torch.LongTensor(timestamps),
                torch.LongTensor(uids),
                torch.LongTensor(quadkey_indices),
                torch.LongTensor(tile_ids),
                torch.LongTensor(answer),
                torch.LongTensor(tile_labels),
                torch.FloatTensor(coords))


class Data_Val:
    def __init__(self, data_train, data_val, args, quadkey_vocab, tiles, tile_vocab_size, tile_to_poi, poi_to_tile, smap_reverse):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2answer = data_val
        self.max_len = args.max_len
        self.quadkey_vocab = quadkey_vocab
        self.tiles = tiles
        self.tile_vocab_size = tile_vocab_size
        self.tile_to_poi = tile_to_poi
        self.poi_to_tile = poi_to_tile
        self.smap_reverse = smap_reverse
        self.dataset_name = args.dataset
        self.cache_dir = './cache'

    def get_pytorch_dataloaders(self):
        dataset = ValDataset(self.u2seq, self.u2answer, self.max_len, self.quadkey_vocab, self.tiles, self.tile_vocab_size, self.tile_to_poi, self.poi_to_tile, smap_reverse=self.smap_reverse, cache_dir=self.cache_dir, dataset_name=self.dataset_name)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, collate_fn=self.collate_fn)
        return dataloader

    def collate_fn(self, batch):
        items, timestamps, uids, quadkey_indices, tile_ids, labels, tile_labels, coords = zip(*batch)
        return (torch.stack(items),
                torch.stack(timestamps),
                torch.stack(uids),
                torch.stack(quadkey_indices),
                torch.stack(tile_ids),
                torch.stack(labels),
                torch.stack(tile_labels),
                torch.stack(coords))


class TestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2_seq_add, u2answer, max_len, quadkey_vocab, tiles, tile_vocab_size, tile_to_poi, poi_to_tile, lod=17, smap_reverse=None, cache_dir='./cache', dataset_name='gowalla'):
        self.u2seq = u2seq
        self.u2seq_add = u2_seq_add
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.quadkey_vocab = quadkey_vocab
        self.tiles = tiles
        self.tile_vocab_size = tile_vocab_size
        self.tile_to_poi = tile_to_poi
        self.poi_to_tile = poi_to_tile
        self.lod = lod
        self.smap_reverse = smap_reverse or {}
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name
        # self.precomputed_data = self._precompute_quadkeys_and_tiles()

    def _precompute_quadkeys_and_tiles(self):
        """预计算所有序列的 quadkeys 和 tile_ids"""
        precomputed = {}
        for idx, user in enumerate(self.users):
            seq = self.u2seq[user]
            seq = [[self.smap_reverse.get(item[0], -1), int(item[1].timestamp()), item[2], item[3], item[4]] for item in
                   seq]
            seq = seq[-self.max_len:]
            padding_len = self.max_len - len(seq)
            if padding_len > 0:
                padding_len = padding_len - 1
            else:
                seq = seq[1:]
            seq = [[0, 0, 0, 0.0, 0.0]] * padding_len + seq + [
                [0, int(self.u2answer[user][0][1].timestamp()), 0, 0.0, 0.0]]

            quadkeys = []
            tile_ids = []
            coords = []
            for item in seq:
                lat, lon = item[3], item[4]
                if lat == 0.0 and lon == 0.0:
                    quadkeys.append(['0'])
                    tile_ids.append(0)
                    coords.append([0.0, 0.0])
                else:
                    qk = Quadkey.from_geo((lat, lon), self.lod)
                    qk_str = str(qk)
                    qk_ngrams = ' '.join([''.join(x) for x in ngrams(qk_str, 6)]) if len(qk_str) >= 6 else qk_str
                    quadkeys.append(qk_ngrams.split())
                    tile_id = map_to_tile(self.tiles, lon, lat)
                    tile_id = tile_id if tile_id != -1 and tile_id < self.tile_vocab_size else self.tile_vocab_size - 1
                    tile_ids.append(tile_id)
                    coords.append([lat, lon])
            precomputed[idx] = (quadkeys, tile_ids, coords)
        return precomputed

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        raw_answer = self.u2answer[user][0][0]
        answer = self.smap_reverse.get(raw_answer, -1)
        answer = [answer]
        unk_tile_id = self.tile_vocab_size - 1
        tile_label = self.poi_to_tile.get(answer[0], unk_tile_id)
        tile_labels = [tile_label]
        last_time = self.u2answer[user][0][1]
        last_uid = self.u2answer[user][0][2]
        seq = [[self.smap_reverse.get(item[0], -1), int(item[1].timestamp()), item[2], item[3], item[4]] for item in seq]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        if padding_len > 0:
            padding_len = padding_len - 1
        else:
            seq = seq[1:]
        seq = [[0, 0, 0, 0.0, 0.0]] * padding_len + seq + [[0, int(last_time.timestamp()), last_uid, 0.0, 0.0]]
        items = [x[0] for x in seq]
        timestamps = [x[1] for x in seq]
        uids = [x[2] for x in seq]
        # quadkeys, tile_ids, coords = self.precomputed_data[index]
        coords = [[x[3], x[4]] for x in seq]
        tile_ids = [self.poi_to_tile.get(key) for key in items]
        # unk_index = self.quadkey_vocab['<unk>']
        # quadkey_indices = [[self.quadkey_vocab[token] if token in self.quadkey_vocab else unk_index for token in qk] for qk in quadkeys]
        # max_ngram_len = max(len(indices) for indices in quadkey_indices)
        # pad_index = self.quadkey_vocab['<pad>']
        # quadkey_indices = [indices + [pad_index] * (max_ngram_len - len(indices)) for indices in quadkey_indices]
        quadkey_indices = []
        return (torch.LongTensor(items),
                torch.LongTensor(timestamps),
                torch.LongTensor(uids),
                torch.LongTensor(quadkey_indices),
                torch.LongTensor(tile_ids),
                torch.LongTensor(answer),
                torch.LongTensor(tile_labels),
                torch.FloatTensor(coords))


class Data_Test:
    def __init__(self, data_train, data_val, data_test, args, quadkey_vocab, tiles, tile_vocab_size, tile_to_poi, poi_to_tile, smap_reverse):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2seq_add = data_val
        self.u2answer = data_test
        self.max_len = args.max_len
        self.quadkey_vocab = quadkey_vocab
        self.tiles = tiles
        self.tile_vocab_size = tile_vocab_size
        self.tile_to_poi = tile_to_poi
        self.poi_to_tile = poi_to_tile
        self.smap_reverse = smap_reverse or {}
        self.dataset_name = args.dataset
        self.cache_dir = './cache'

    def get_pytorch_dataloaders(self):
        dataset = TestDataset(self.u2seq, self.u2seq_add, self.u2answer, self.max_len, self.quadkey_vocab, self.tiles, self.tile_vocab_size, self.tile_to_poi, self.poi_to_tile, smap_reverse=self.smap_reverse, cache_dir=self.cache_dir, dataset_name=self.dataset_name)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, collate_fn=self.collate_fn)
        return dataloader

    def collate_fn(self, batch):
        items, timestamps, uids, quadkey_indices, tile_ids, labels, tile_labels, coords = zip(*batch)
        return (torch.stack(items),
                torch.stack(timestamps),
                torch.stack(uids),
                torch.stack(quadkey_indices),
                torch.stack(tile_ids),
                torch.stack(labels),
                torch.stack(tile_labels),
                torch.stack(coords))


class CHLSDataset(data_utils.Dataset):
    def __init__(self, data, max_len, quadkey_vocab, tiles, tile_vocab_size, tile_to_poi, poi_to_tile, lod=17, smap_reverse=None, cache_dir='./cache', dataset_name='gowalla'):
        self.data = data
        self.max_len = max_len
        self.quadkey_vocab = quadkey_vocab
        self.tiles = tiles
        self.tile_vocab_size = tile_vocab_size
        self.tile_to_poi = tile_to_poi
        self.poi_to_tile = poi_to_tile
        self.lod = lod
        self.smap_reverse = smap_reverse or {}
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name
        self.precomputed_data = self._precompute_quadkeys_and_tiles()

    def _precompute_quadkeys_and_tiles(self):
        """预计算所有序列的 quadkeys 和 tile_ids"""
        precomputed = {}
        for idx in range(len(self.data)):
            data_temp = self.data[idx]
            seq = data_temp[:-1]
            seq = [[self.smap_reverse.get(item[0], -1), int(item[1].timestamp()), item[2], item[3], item[4]] for item in
                   seq]
            seq = seq[-self.max_len:]
            padding_len = self.max_len - len(seq)
            if padding_len > 0:
                padding_len = padding_len - 1
            else:
                seq = seq[1:]
            seq = [[0, 0, 0, 0.0, 0.0]] * padding_len + seq + [[0, int(data_temp[-1][1].timestamp()), 0, 0.0, 0.0]]

            quadkeys = []
            tile_ids = []
            coords = []
            for item in seq:
                lat, lon = item[3], item[4]
                if lat == 0.0 and lon == 0.0:
                    quadkeys.append(['0'])
                    tile_ids.append(0)
                    coords.append([0.0, 0.0])
                else:
                    qk = Quadkey.from_geo((lat, lon), self.lod)
                    qk_str = str(qk)
                    qk_ngrams = ' '.join([''.join(x) for x in ngrams(qk_str, 6)]) if len(qk_str) >= 6 else qk_str
                    quadkeys.append(qk_ngrams.split())
                    tile_id = map_to_tile(self.tiles, lon, lat)
                    tile_id = tile_id if tile_id != -1 and tile_id < self.tile_vocab_size else self.tile_vocab_size - 1
                    tile_ids.append(tile_id)
                    coords.append([lat, lon])
            precomputed[idx] = (quadkeys, tile_ids, coords)
        return precomputed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_temp = self.data[index]
        seq = data_temp[:-1]
        raw_answer = data_temp[-1][0]  # 原始 POI ID，例如 1001
        answer = self.smap_reverse.get(raw_answer, -1)  # 映射为连续 ID，例如 1001 -> 1
        answer = [answer]

        # 生成 tile_label
        unk_tile_id = self.tile_vocab_size - 1  # <unk> 瓦片ID
        tile_label = self.poi_to_tile.get(answer[0], unk_tile_id)  # 直接查找 POI 到瓦片的映射
        tile_labels = [tile_label]

        last_time = data_temp[-1][1]
        last_uid = data_temp[-1][2]
        seq = [[self.smap_reverse.get(item[0], -1), int(item[1].timestamp()), item[2], item[3], item[4]] for item in seq]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        if padding_len > 0:
            padding_len = padding_len - 1
        else:
            seq = seq[1:]
        seq = [[0, 0, 0, 0.0, 0.0]] * padding_len + seq + [[0, int(last_time.timestamp()), last_uid, 0.0, 0.0]]

        items = [x[0] for x in seq]
        timestamps = [x[1] for x in seq]
        uids = [x[2] for x in seq]

        quadkeys, tile_ids, coords = self.precomputed_data[index]
        unk_index = self.quadkey_vocab['<unk>']
        quadkey_indices = [[self.quadkey_vocab[token] if token in self.quadkey_vocab else unk_index for token in qk] for
                           qk in quadkeys]
        max_ngram_len = max(len(indices) for indices in quadkey_indices)
        pad_index = self.quadkey_vocab['<pad>']
        quadkey_indices = [indices + [pad_index] * (max_ngram_len - len(indices)) for indices in quadkey_indices]

        return (torch.LongTensor(items),
                torch.LongTensor(timestamps),
                torch.LongTensor(uids),
                torch.LongTensor(quadkey_indices),
                torch.LongTensor(tile_ids),
                torch.LongTensor(answer),
                torch.LongTensor(tile_labels),
                torch.LongTensor(coords))


class Data_CHLS:
    def __init__(self, data, args, quadkey_vocab, tiles, tile_vocab_size, tile_to_poi, poi_to_tile, smap_reverse):
        self.batch_size = args.batch_size
        self.max_len = args.max_len
        self.data = data
        self.quadkey_vocab = quadkey_vocab
        self.tiles = tiles
        self.tile_vocab_size = tile_vocab_size
        self.tile_to_poi = tile_to_poi
        self.poi_to_tile = poi_to_tile
        self.smap_reverse = smap_reverse or {}
        self.dataset_name = args.dataset
        self.cache_dir = './cache'

    def get_pytorch_dataloaders(self):
        dataset = CHLSDataset(self.data, self.max_len, self.quadkey_vocab, self.tiles, self.tile_vocab_size, self.tile_to_poi, self.poi_to_tile, smap_reverse=self.smap_reverse, cache_dir=self.cache_dir, dataset_name=self.dataset_name)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True,collate_fn=self.collate_fn)
        return dataloader

    def collate_fn(self, batch):
        items, timestamps, uids, quadkey_indices, tile_ids, labels, tile_labels, coords = zip(*batch)
        return (torch.stack(items),
                torch.stack(timestamps),
                torch.stack(uids),
                torch.stack(quadkey_indices),
                torch.stack(tile_ids),
                torch.stack(labels),
                torch.stack(tile_labels),
                torch.stack(coords))


def get_norm_time96(time):
    hour = time.hour
    minute = time.minute
    ans = minute//15 + 4*hour
    return ans


def get_day_norm7(time):
    day_number = time.weekday()
    return day_number