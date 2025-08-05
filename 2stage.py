import torch
import torch.nn as nn
import torch.optim as optim

class TwoStagePOITrainer:
    def __init__(self, model, tile_lr=1e-3, poi_lr=1e-3, mode='pretrain_tile'):
        self.model = model
        self.mode = mode  # 'pretrain_tile', 'pretrain_poi', 'joint'

        # 分别定义两组参数
        self.optimizer = optim.Adam([
            {'params': model.diffu_tile.parameters(), 'lr': tile_lr},
            {'params': model.pdiffu_poi.parameters(), 'lr': poi_lr}
        ])

        self.loss_fn = nn.CrossEntropyLoss()

    def train_step(self, rep_tile, rep_poi, tile_labels, poi_labels):
        self.model.train()
        self.optimizer.zero_grad()

        # 获取损失及得分
        tile_loss, poi_loss, joint_loss, *_ = self.model.loss_two_stage(rep_tile, rep_poi, tile_labels, poi_labels)

        if self.mode == 'pretrain_tile':
            loss = tile_loss
        elif self.mode == 'pretrain_poi':
            loss = poi_loss
        elif self.mode == 'joint':
            loss = tile_loss + joint_loss
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        loss.backward()
        self.optimizer.step()
        return {
            'tile_loss': tile_loss.item(),
            'poi_loss': poi_loss.item()
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
        self.current_mode = 'pretrain_tile'
        self.switch_epoch_1 = None
        self.switch_epoch_2 = None

    def update_and_check(self, trainer, epoch, tile_loss, poi_loss):
        if self.current_mode == 'pretrain_tile':
            self.tile_history.append(tile_loss)
            if len(self.tile_history) >= self.tile_patience:
                recent = self.tile_history[-self.tile_patience:]
                if all(l < self.tile_threshold for l in recent):
                    trainer.switch_mode('pretrain_poi')
                    self.current_mode = 'pretrain_poi'
                    self.switch_epoch_1 = epoch
                    print(f"[Monitor] Switched to pretrain_poi at epoch {epoch}")

        elif self.current_mode == 'pretrain_poi':
            self.poi_history.append(poi_loss)
            if len(self.poi_history) >= self.poi_patience:
                recent = self.poi_history[-self.poi_patience:]
                if all(l < self.poi_threshold for l in recent):
                    trainer.switch_mode('joint')
                    self.current_mode = 'joint'
                    self.switch_epoch_2 = epoch
                    print(f"[Monitor] Switched to joint at epoch {epoch}")



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


