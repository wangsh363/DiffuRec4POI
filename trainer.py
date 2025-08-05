import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import numpy as np
import copy
import os
import time
import pickle

class TwoStagePOITrainer:
    def __init__(self, model, mode='pretrain'):
        self.model = model
        self.mode = mode  # 'pretrain_tile', 'pretrain_poi', 'joint'

    def train_step(self, rep_tile, rep_poi, tile_labels, poi_labels):

        # 获取损失及得分
        tile_loss, poi_loss, joint_loss, *_ = self.model.loss_two_stage(rep_tile, rep_poi, tile_labels, poi_labels)

        if self.mode == 'pretrain':
            loss = tile_loss + poi_loss
            loss.backward()
        elif self.mode == 'pretrain_poi':
            loss = poi_loss
            loss.backward()
        elif self.mode == 'pretrain_tile':
            loss = tile_loss
            loss.backward()
        elif self.mode == 'joint':
            loss = joint_loss
            loss.backward()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # loss.backward()
        # self.optimizer_tile.step()
        # self.optimizer_poi.step()
        return {
            'tile_loss': 0.0 if self.mode == 'pretrain_poi' else tile_loss.item(),
            'poi_loss': 0.0 if self.mode == 'pretrain_tile' else poi_loss.item(),
            'joint_loss': 0.0 if self.mode != 'joint' else joint_loss.item()
        }

    def switch_mode(self, mode):
        assert mode in ['pretrain_tile', 'pretrain_poi', 'joint']
        self.mode = mode
        print(f"Switched training mode to: {mode}")


class DynamicLossMonitor:
    def __init__(self, 
                 tile_patience=3, tile_threshold=0.2,
                 poi_patience=3, poi_threshold=0.2):
        self.tile_patience = tile_patience
        self.tile_threshold = tile_threshold
        self.poi_patience = poi_patience
        self.poi_threshold = poi_threshold

        self.tile_history = []
        self.poi_history = []
        self.tile_done = False
        self.poi_done = False
        self.current_mode = 'pretrain'  # new initial mode
        self.switch_epoch_tile = None
        self.switch_epoch_poi = None
        self.switch_epoch_joint = None

    def update_and_check(self, trainer, epoch, logger, tile_loss, poi_loss):
        self.tile_history.append(tile_loss)
        self.poi_history.append(poi_loss)

        # Check tile warm-up done
        if not self.tile_done and len(self.tile_history) >= self.tile_patience:
            recent_tile = self.tile_history[-self.tile_patience:]
            if all(l < self.tile_threshold for l in recent_tile):
                self.tile_done = True
                self.switch_epoch_tile = epoch
                print(f"[Monitor] Tile warm-up done at epoch {epoch}")
                logger.info(f"[Monitor] Tile warm-up done at epoch {epoch}")

        # Check poi warm-up done
        if not self.poi_done and len(self.poi_history) >= self.poi_patience:
            recent_poi = self.poi_history[-self.poi_patience:]
            if all(l < self.poi_threshold for l in recent_poi):
                self.poi_done = True
                self.switch_epoch_poi = epoch
                print(f"[Monitor] POI warm-up done at epoch {epoch}")
                logger.info(f"[Monitor] POI warm-up done at epoch {epoch}")

        # Determine next mode
        if self.current_mode == 'pretrain':
            if self.tile_done and not self.poi_done:
                self.current_mode = 'pretrain_poi'
                trainer.switch_mode('pretrain_poi')
                print(f"[Monitor] Switched to pretrain_poi at epoch {epoch}")
                logger.info(f"[Monitor] Switched to pretrain_poi at epoch {epoch}")
            elif self.poi_done and not self.tile_done:
                self.current_mode = 'pretrain_tile'
                trainer.switch_mode('pretrain_tile')
                print(f"[Monitor] Switched to pretrain_tile at epoch {epoch}")
                logger.info(f"[Monitor] Switched to pretrain_tile at epoch {epoch}")
            elif self.tile_done and self.poi_done:
                self.current_mode = 'joint'
                trainer.switch_mode('joint')
                self.switch_epoch_joint = epoch
                print(f"[Monitor] Switched to joint at epoch {epoch}")
                logger.info(f"[Monitor] Switched to joint at epoch {epoch}")

        elif self.current_mode in ['pretrain_tile', 'pretrain_poi']:
            if self.tile_done and self.poi_done:
                self.current_mode = 'joint'
                trainer.switch_mode('joint')
                self.switch_epoch_joint = epoch
                print(f"[Monitor] Switched to joint at epoch {epoch}")
                logger.info(f"[Monitor] Switched to joint at epoch {epoch}")


def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError


def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return hr


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_hr(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['HR@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics


def hrs_and_ndcgs_k_from_indices(top_k_indices, labels, ks):
    metrics = {}
    labels = labels.clone().detach().to('cpu').view(-1)
    top_k_indices = top_k_indices.clone().detach().to('cpu')
    batch_size = labels.size(0)

    # 计算HR
    hr = []
    labels_squeezed = labels.squeeze(-1)  # (batch_size,)
    for k in ks:
        top_k = top_k_indices[:, :min(k, top_k_indices.size(1))]  # (batch_size, k)
        hit = (top_k == labels.unsqueeze(1)).any(dim=1)  # (batch_size,)
        hr_val = hit.float().mean().item()
        hr.append(hr_val)

    # 计算NDCG
    ndcg = []
    max_ks = max(ks)
    hit = (labels.unsqueeze(1) == top_k_indices).int()  # (batch_size, max_ks)
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k - 1)))  # 理想 DCG
        predict_dcg = dcg(hit[:, :k])  # 预测 DCG
        ndcg.append((predict_dcg / max_dcg).mean().item())

    for k, hr_val, ndcg_val in zip(ks, hr, ndcg):
        metrics['HR@%d' % k] = hr_val
        metrics['NDCG@%d' % k] = ndcg_val
    return metrics


def LSHT_inference(model_joint, args, data_loader):
    device = args.device
    model_joint = model_joint.to(device)
    unk_poi_id = args.item_num - 1  # <unk> POI ID
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in data_loader:
            test_batch = [x.to(device) for x in test_batch]

            # 解包输入数据，包括 tile_label
            items, timestamps, uids, quadkeys, tiles, labels, tile_labels, coords = test_batch
            sequence = (items, timestamps, uids, quadkeys, tiles)

            # 推理模式：获取Top K瓦片和Top K POI
            top_k_tiles, top_k_pois, (tile_time_target, poi_time_target) = model_joint(sequence, labels, tile_labels, train_flag=False, coords=coords)

            # top_k_pois 已经是Top K的POI推荐列表，直接用于评估
            valid_mask = labels.squeeze(-1) != unk_poi_id
            if not valid_mask.all():
                print(f"警告: 数据集中包含 {valid_mask.size(0) - valid_mask.sum().item()} 个 <unk> 标签")
            top_k_pois = top_k_pois[valid_mask]
            labels = labels[valid_mask]
            if top_k_pois.size(0) > 0:
                metrics = hrs_and_ndcgs_k_from_indices(top_k_pois, labels, [5, 10, 20])
                for k, v in metrics.items():
                    test_metrics_dict[k].append(v)

    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print("Inference Results:")
    print(test_metrics_dict_mean)


def model_train(tra_data_loader, val_data_loader, test_data_loader, model_joint, args, logger):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    model_joint = model_joint.to(device)
    is_parallel = args.num_gpu > 1
    if is_parallel:
        model_joint = nn.DataParallel(model_joint)
    # optimizer = optimizers(model_joint, args)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    # optimizer_tile = optim.Adam(model_joint.diffu_tile.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_poi  = optim.Adam(model_joint.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler_tile = optim.lr_scheduler.StepLR(optimizer_tile, step_size=args.decay_step, gamma=args.gamma)
    scheduler_poi = optim.lr_scheduler.StepLR(optimizer_poi, step_size=args.decay_step, gamma=args.gamma)
    best_metrics_dict = {'Best_HR@5': 0, 'Best_NDCG@5': 0, 'Best_HR@10': 0, 'Best_NDCG@10': 0, 'Best_HR@20': 0, 'Best_NDCG@20': 0}
    best_epoch = {'Best_epoch_HR@5': 0, 'Best_epoch_NDCG@5': 0, 'Best_epoch_HR@10': 0, 'Best_epoch_NDCG@10': 0, 'Best_epoch_HR@20': 0, 'Best_epoch_NDCG@20': 0}
    best_metrics_dict_tile = {'tile_Best_HR@5': 0, 'tile_Best_NDCG@5': 0, 'tile_Best_HR@10': 0, 'tile_Best_NDCG@10': 0, 'tile_Best_HR@20': 0, 'tile_Best_NDCG@20': 0}
    best_epoch_tile = {'tile_Best_epoch_HR@5': 0, 'tile_Best_epoch_NDCG@5': 0, 'tile_Best_epoch_HR@10': 0, 'tile_Best_epoch_NDCG@10': 0, 'tile_Best_epoch_HR@20': 0, 'tile_Best_epoch_NDCG@20': 0}
    best_metrics_dict_poi = {'poi_Best_HR@5': 0, 'poi_Best_NDCG@5': 0, 'poi_Best_HR@10': 0, 'poi_Best_NDCG@10': 0, 'poi_Best_HR@20': 0, 'poi_Best_NDCG@20': 0}
    best_epoch_poi = {'poi_Best_epoch_HR@5': 0, 'poi_Best_epoch_NDCG@5': 0, 'poi_Best_epoch_HR@10': 0, 'poi_Best_epoch_NDCG@10': 0, 'poi_Best_epoch_HR@20': 0, 'poi_Best_epoch_NDCG@20': 0}
    bad_count = 0
    unk_poi_id = 0
    unk_tile_id = 0
    monitor = DynamicLossMonitor(
    tile_patience=3, tile_threshold=1.3,
    poi_patience=3, poi_threshold=2.1
    )
    trainer = TwoStagePOITrainer(model_joint, mode='pretrain')

    for epoch_temp in range(epochs):
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        model_joint.train()
        flag_update = 0
        for index_temp, train_batch in enumerate(tra_data_loader):
            train_batch = [x.to(device) for x in train_batch]
            items, timestamps, uids, quadkeys, tiles, labels, tile_labels, coords, tile_coords = train_batch
            sequence = (items, timestamps, uids, quadkeys, tiles)
            if labels.max().item() >= args.item_num + 1 or labels.min().item() < 0:
                print(f"警告: 无效 POI labels 检测到，min={labels.min().item()}, max={labels.max().item()}, item_num={args.item_num}")
                # labels = torch.where(labels == -1, unk_poi_id, labels)
                if labels.max().item() >= args.item_num + 1:
                    print(f"错误: POI 标签仍然无效，max={labels.max().item()}")
                    # labels = torch.clamp(labels, 0, unk_poi_id)
            if tile_labels.max().item() >= model_joint.tile_vocab_size or tile_labels.min().item() < 0:
                print(f"警告: 无效 tile_labels 检测到，min={tile_labels.min().item()}, max={tile_labels.max().item()}, tile_vocab_size={model_joint.tile_vocab_size}")
                # tile_labels = torch.where(tile_labels == -1, unk_tile_id, tile_labels)
                if tile_labels.max().item() >= model_joint.tileVocab_size:
                    print(f"错误: 瓦片标签仍然无效，max={tile_labels.max().item()}")
                    # tile_labels = torch.clamp(tile_labels, 0, unk_tile_id)
            if items.max().item() >= args.item_num + 1 or items.min().item() < 0:
                print(f"警告: 无效 items 检测到，min={items.min().item()}, max={items.max().item()}, item_num={args.item_num}")
                # items = torch.clamp(items, 0, unk_poi_id)
            # optimizer.zero_grad()
            # optimizer_tile.zero_grad()
            optimizer_poi.zero_grad()
            condition, diffu_rep, weights, t, item_rep_dis, seq_rep_dis, time_target = model_joint(sequence, labels, tile_labels, train_flag=True, coords=tile_coords)
            tile_rep_diffu, poi_rep_diffu = diffu_rep
            tile_weights, poi_weights = weights
            tile_t, poi_t = t
            tile_time_target, poi_time_target = time_target
            # loss_diffu_tile = model_joint.loss_arcface(tile_rep_diffu, tile_labels, target_type="tile")
            # loss_diffu_tile = model_joint.loss_diffu_ce(tile_rep_diffu, tile_labels)
            # loss_diffu_poi = model_joint.loss_diffu_ce(poi_rep_diffu, labels)
            # loss_diffu_tile, loss_diffu_poi, _, _, _, _ = model_joint.loss_two_stage(tile_rep_diffu, poi_rep_diffu, tile_labels, labels)
            # loss_all = loss_diffu_tile + loss_diffu_poi
            # loss_all = loss_diffu_tile + loss_diffu_poi
            logs = trainer.train_step(tile_rep_diffu, poi_rep_diffu, tile_labels, labels)
            loss_diffu_tile = logs['tile_loss']
            loss_diffu_poi = logs['poi_loss']
            loss_all = logs['joint_loss']
            # loss_all.backward()
            # optimizer.step()
            # if trainer.mode == 'pretrain_tile':
            #     optimizer_tile.step()
            # elif trainer.mode == 'pretrain_poi':
            #     optimizer_poi.step()
            # elif trainer.mode == 'joint':
            #     optimizer_tile.step()
            #     optimizer_poi.step()
            optimizer_poi.step()
            if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
                print('[%d/%d] Loss_all: %.4f Loss_tile: %.4f Loss_poi: %.4f' % (index_temp, len(tra_data_loader), loss_all, loss_diffu_tile, loss_diffu_poi))
                logger.info('[%d/%d] Loss_all: %.4f Loss_tile: %.4f Loss_poi: %.4f' % (index_temp, len(tra_data_loader), loss_all, loss_diffu_tile, loss_diffu_poi))

        monitor.update_and_check(
            trainer, epoch_temp, logger,
            tile_loss=logs['tile_loss'],
            poi_loss=logs['poi_loss']
        )
        print("loss in epoch {}: {}".format(epoch_temp, loss_all))
        # if trainer.mode != 'pretrain_tile':
        #     scheduler_poi.step()
        # elif trainer.mode != 'pretrain_poi':
        #     scheduler_tile.step()
        scheduler_poi.step()
        # lr_scheduler.step()
        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))
            model_joint.eval()
            with torch.no_grad():
                metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
                metrics_dict_poi = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
                metrics_dict_tile = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
                for val_batch in val_data_loader:
                    val_batch = [x.to(device) for x in val_batch]
                    items, timestamps, uids, quadkeys, tiles, labels, tile_labels, coords, tile_coords = val_batch
                    sequence = (items, timestamps, uids, quadkeys, tiles)
                    poi_rep, tile_rep = model_joint(sequence, labels, tile_labels, train_flag=False, coords=tile_coords)
                    # valid_mask = labels.squeeze(-1) != unk_poi_id
                    # valid_mask_tile = tile_labels.squeeze(-1) != unk_tile_id
                    # if not valid_mask.all():
                    #     print(f"警告: 验证集中包含 {valid_mask.size(0) - valid_mask.sum().item()} 个 <unk> 标签")
                    # top_k_pois = top_k_pois[valid_mask]
                    # labels = labels[valid_mask]
                    # top_k_tiles = top_k_tiles[valid_mask_tile]
                    # tile_labels = tile_labels[valid_mask_tile]

                    # scores_poi = model_joint.diffu_rep_pre(poi_rep)
                    _, _, _, scores_tile, scores_poi, final_scores_poi = model_joint.loss_two_stage(tile_rep, poi_rep, tile_labels, labels)
                    metrics_poi = hrs_and_ndcgs_k(scores_poi, labels, metric_ks)
                    for k, v in metrics_poi.items():
                        metrics_dict_poi[k].append(v)

                    metrics = hrs_and_ndcgs_k(final_scores_poi, labels, metric_ks)
                    for k, v in metrics.items():
                        metrics_dict[k].append(v)

                    metrics_tile = hrs_and_ndcgs_k(scores_tile, tile_labels, metric_ks)
                    for k, v in metrics_tile.items():
                        metrics_dict_tile[k].append(v)
            
            # for key_temp, values_temp in metrics_dict_poi.items():
            #     values_mean = round(np.mean(values_temp) * 100, 4)
            #     if values_mean > best_metrics_dict_poi['poi_Best_' + key_temp]:
            #         flag_update_poi = 1
            #         bad_count_poi = 0
            #         best_metrics_dict_poi['poi_Best_' + key_temp] = values_mean
            #         best_epoch_poi['poi_Best_epoch_' + key_temp] = epoch_temp

            # for key_temp, values_temp in metrics_dict.items():
            #     values_mean = round(np.mean(values_temp) * 100, 4)
            #     if values_mean > best_metrics_dict['Best_' + key_temp]:
            #         flag_update = 1
            #         bad_count = 0
            #         best_metrics_dict['Best_' + key_temp] = values_mean
            #         best_epoch['Best_epoch_' + key_temp] = epoch_temp

            # for key_temp, values_temp in metrics_dict_tile.items():
            #     values_mean = round(np.mean(values_temp) * 100, 4)
            #     if values_mean > best_metrics_dict_tile['tile_Best_' + key_temp]:
            #         flag_update_tile = 1
            #         bad_count_tile = 0
            #         best_metrics_dict_tile['tile_Best_' + key_temp] = values_mean
            #         best_epoch_tile['tile_Best_epoch_' + key_temp] = epoch_temp
            
            if trainer.mode != 'pretrain_tile':
                for key_temp, values_temp in metrics_dict_poi.items():
                    values_mean = round(np.mean(values_temp) * 100, 4)
                    if values_mean > best_metrics_dict_poi['poi_Best_' + key_temp]:
                        flag_update_poi = 1
                        bad_count_poi = 0
                        best_metrics_dict_poi['poi_Best_' + key_temp] = values_mean
                        best_epoch_poi['poi_Best_epoch_' + key_temp] = epoch_temp
                if flag_update_poi == 0:
                    bad_count_poi += 1
                else:
                    print(best_metrics_dict_poi)
                    print(best_epoch_poi)
                    logger.info(best_metrics_dict_poi)
                    logger.info(best_epoch_poi)

            if trainer.mode == 'joint':
                for key_temp, values_temp in metrics_dict.items():
                    values_mean = round(np.mean(values_temp) * 100, 4)
                    if values_mean > best_metrics_dict['Best_' + key_temp]:
                        flag_update = 1
                        bad_count = 0
                        best_metrics_dict['Best_' + key_temp] = values_mean
                        best_epoch['Best_epoch_' + key_temp] = epoch_temp
                if flag_update == 0:
                    bad_count += 1
                else:
                    print(best_metrics_dict)
                    print(best_epoch)
                    logger.info(best_metrics_dict)
                    logger.info(best_epoch)
                    best_model = copy.deepcopy(model_joint)
                if bad_count >= args.patience:
                    break

            if trainer.mode != 'pretrain_poi':
                for key_temp, values_temp in metrics_dict_tile.items():
                    values_mean = round(np.mean(values_temp) * 100, 4)
                    if values_mean > best_metrics_dict_tile['tile_Best_' + key_temp]:
                        flag_update_tile = 1
                        bad_count_tile = 0
                        best_metrics_dict_tile['tile_Best_' + key_temp] = values_mean
                        best_epoch_tile['tile_Best_epoch_' + key_temp] = epoch_temp
                if flag_update_tile == 0:
                    bad_count_tile += 1
                else:
                    print(best_metrics_dict_tile)
                    print(best_epoch_tile)
                    logger.info(best_metrics_dict_tile)
                    logger.info(best_epoch_tile)

    logger.info(best_metrics_dict)
    logger.info(best_epoch)
    if args.eval_interval > epochs:
        best_model = copy.deepcopy(model_joint)
    top_100_item = []
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_poi = {'poi_HR@5': [], 'poi_NDCG@5': [], 'poi_HR@10': [], 'poi_NDCG@10': [], 'poi_HR@20': [], 'poi_NDCG@20': []}
        test_metrics_dict_tile = {'tile_HR@5': [], 'tile_NDCG@5': [], 'tile_HR@10': [], 'tile_NDCG@10': [], 'tile_HR@20': [], 'tile_NDCG@20': []}
        test_metrics_dict_poi_mean = {}
        test_metrics_dict_tile_mean = {}
        test_metrics_dict_mean = {}
        for test_batch in test_data_loader:
            test_batch = [x.to(device) for x in test_batch]
            items, timestamps, uids, quadkeys, tiles, labels, tile_labels, coords, tile_coords = test_batch
            sequence = (items, timestamps, uids, quadkeys, tiles)
            poi_rep, tile_rep = best_model(sequence, labels, tile_labels, train_flag=False, coords=tile_coords)
            # valid_mask = labels.squeeze(-1) != unk_poi_id
            # if not valid_mask.all():
            #     print(f"警告: 测试集中包含 {valid_mask.size(0) - valid_mask.sum().item()} 个 <unk> 标签")
            # top_k_pois = top_k_pois[valid_mask]
            # labels = labels[valid_mask]
            _, _, _, scores_tile, scores_poi, final_scores_poi = best_model.loss_two_stage(tile_rep, poi_rep, tile_labels, labels)
            metrics = hrs_and_ndcgs_k(final_scores_poi, labels, metric_ks)
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
            metrics_poi = hrs_and_ndcgs_k(scores_poi, labels, metric_ks)
            for k, v in metrics_poi.items():
                test_metrics_dict_poi[k].append(v)
            metrics_tile = hrs_and_ndcgs_k(scores_tile, tile_labels, metric_ks)
            for k, v in metrics_tile.items():
                test_metrics_dict_tile[k].append(v)
        for key_temp, values_temp in test_metrics_dict.items():
            values_mean = round(np.mean(values_temp) * 100, 4)
            test_metrics_dict_mean[key_temp] = values_mean
        for key_temp, values_temp in test_metrics_dict_poi.items():
            values_mean = round(np.mean(values_temp) * 100, 4)
            test_metrics_dict_poi_mean[key_temp] = values_mean
        for key_temp, values_temp in test_metrics_dict_tile.items():
            values_mean = round(np.mean(values_temp) * 100, 4)
            test_metrics_dict_tile_mean[key_temp] = values_mean
        print('Test------------------------------------------------------')
        logger.info('Test------------------------------------------------------')
        print(test_metrics_dict_mean)
        logger.info(test_metrics_dict_mean)
        print(test_metrics_dict_poi_mean)
        logger.info(test_metrics_dict_poi_mean)
        print(test_metrics_dict_tile_mean)
        logger.info(test_metrics_dict_tile_mean)
        print('Best Eval---------------------------------------------------------')
        logger.info('Best Eval---------------------------------------------------------')
        # print(best_metrics_dict)
        # print(best_epoch)
        # logger.info(best_metrics_dict)
        # logger.info(best_epoch)
        print(args)

        print('saving model...')
        # 模型保存目录
        save_dir = 'model' 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 保存最好的模型参数
        best_model_path = os.path.join(save_dir, 'best_model_' + args.dataset + '.pth')
        torch.save(best_model.state_dict(), best_model_path)
        logger.info(f"Best model saved at {best_model_path}")


        if args.diversity_measure:
            path_data = '../datasets/data/category/' + args.dataset + '/id_category_dict.pkl'
            with open(path_data, 'rb') as f:
                id_category_dict = pickle.load(f)
            id_top_100 = torch.cat(top_100_item, dim=0).tolist()
            category_list_100 = []
            for id_top_100_temp in id_top_100:
                category_temp_list = []
                for id_temp in id_top_100_temp:
                    if id_temp != unk_poi_id:
                        category_temp_list.append(id_category_dict[id_temp])
                category_list_100.append(category_temp_list)
            path_data_category = '../datasets/data/category/' + args.dataset + '/DiffuRec_top100_category.pkl'
            with open(path_data_category, 'wb') as f:
                pickle.dump(category_list_100, f)

    return best_model, test_metrics_dict_mean