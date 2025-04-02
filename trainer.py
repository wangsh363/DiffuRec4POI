import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import numpy as np
import copy
import time
import pickle


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


def LSHT_inference(model_joint, args, data_loader):
    device = args.device
    model_joint = model_joint.to(device)
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in data_loader:
            test_batch = [x.to(device) for x in test_batch]
            
            scores_rec, rep_diffu, _, _, _, _ = model_joint(test_batch[0], test_batch[1], train_flag=False)
            scores_rec_diffu = model_joint.diffu_rep_pre(rep_diffu)
            metrics = hrs_and_ndcgs_k(scores_rec_diffu, test_batch[1], [5, 10, 20])
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print(test_metrics_dict_mean)


def model_train(tra_data_loader, val_data_loader, test_data_loader, model_joint, args, logger):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    model_joint = model_joint.to(device)
    is_parallel = args.num_gpu > 1
    if is_parallel:  # 如果启用数据并行的话，就在多个GPU上进行训练
        model_joint = nn.DataParallel(model_joint)
    optimizer = optimizers(model_joint, args)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    best_metrics_dict = {'Best_HR@5': 0, 'Best_NDCG@5': 0, 'Best_HR@10': 0, 'Best_NDCG@10': 0, 'Best_HR@20': 0, 'Best_NDCG@20': 0}
    best_epoch = {'Best_epoch_HR@5': 0, 'Best_epoch_NDCG@5': 0, 'Best_epoch_HR@10': 0, 'Best_epoch_NDCG@10': 0, 'Best_epoch_HR@20': 0, 'Best_epoch_NDCG@20': 0}
    bad_count = 0


    # # 输出模型的可训练参数表
    # print('注意')
    # print("Model parameters count:", len(list(model_joint.parameters())))
    # for name, param in model_joint.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")



    # 这里是训练的主体
    for epoch_temp in range(epochs):        
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        model_joint.train()
    
        flag_update = 0
        for index_temp, train_batch in enumerate(tra_data_loader):  # 从数据读取器里读取数据
            train_batch = [x.to(device) for x in train_batch]
            # 将每个批次中的数据（如输入和标签）移到指定的设备上，device是GPU或CPU

            optimizer.zero_grad()
            scores, diffu_rep, weights, t, item_rep_dis, seq_rep_dis, time_target = model_joint(train_batch[0], train_batch[1], train_flag=True)
            # 将当前批次的输入数据送入模型 `model_joint`，并获得模型输出
            # `train_batch[0]` 是输入数据，`train_batch[1]` 是目标标签（如分类标签、回归值等）
            # 训练标志 `train_flag=True` 表示这是在训练阶段


            # loss_diffu_value = model_joint.loss_diffu_ce(diffu_rep, train_batch[1])  
            # print(diffu_rep.size(), diffu_rep.dtype)
            # print(train_batch[1].size(), train_batch[1].dtype)
            # print(time_target.size(), time_target.dtype)
            # print(train_batch[1])
            # print("**********************************************")
            # # print(time_target)


            loss_diffu_value = model_joint.loss_diffu_ce(diffu_rep, train_batch[1])   # 目标物品本身加时间

          
            loss_all = loss_diffu_value
            loss_all.backward()

            optimizer.step()

            if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
                print('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))
                logger.info('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))
        print("loss in epoch {}: {}".format(epoch_temp, loss_all.item()))
        lr_scheduler.step()
        # 到这里训练结束了

        # 在模型训练过程中，每隔一定周期（args.eval_interval）对模型进行验证（evaluation）
        # 并计算推荐系统常用的评估指标（如 HR@K 和 NDCG@K）
        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))
            model_joint.eval()  # 切换模型到评估模式。这个是torch模型自己的函数
            with torch.no_grad():  # 禁用梯度计算
                metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
                # metrics_dict_mean = {}
                for val_batch in val_data_loader:
                    val_batch = [x.to(device) for x in val_batch]
                    scores_rec, rep_diffu, _, _, _, _, _ = model_joint(val_batch[0], val_batch[1], train_flag=False)
                    scores_rec_diffu = model_joint.diffu_rep_pre(rep_diffu)    ### inner_production
                    # scores_rec_diffu = model_joint.routing_rep_pre(rep_diffu)   ### routing_rep_pre
                    # 把正确答案提取一下
                    metrics = hrs_and_ndcgs_k(scores_rec_diffu, val_batch[1], metric_ks)
                    for k, v in metrics.items():
                        metrics_dict[k].append(v)


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
            
    
    logger.info(best_metrics_dict)
    logger.info(best_epoch)
        
    if args.eval_interval > epochs:
        best_model = copy.deepcopy(model_joint)
    
    
    top_100_item = []
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in test_data_loader:
            test_batch = [x.to(device) for x in test_batch]
            scores_rec, rep_diffu, _, _, _, _, _ = best_model(test_batch[0], test_batch[1], train_flag=False)
            scores_rec_diffu = best_model.diffu_rep_pre(rep_diffu)   ### Inner Production
            # scores_rec_diffu = best_model.routing_rep_pre(rep_diffu)   ### routing
            
            _, indices = torch.topk(scores_rec_diffu, k=100)
            top_100_item.append(indices)

            metrics = hrs_and_ndcgs_k(scores_rec_diffu, test_batch[1], metric_ks)
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
    
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print('Test------------------------------------------------------')
    logger.info('Test------------------------------------------------------')
    print(test_metrics_dict_mean)
    logger.info(test_metrics_dict_mean)
    print('Best Eval---------------------------------------------------------')
    logger.info('Best Eval---------------------------------------------------------')
    print(best_metrics_dict)
    print(best_epoch)
    logger.info(best_metrics_dict)
    logger.info(best_epoch)

    print(args)

    if args.diversity_measure:
        path_data = '../datasets/data/category/' + args.dataset +'/id_category_dict.pkl'
        with open(path_data, 'rb') as f:
            id_category_dict = pickle.load(f)
        id_top_100 = torch.cat(top_100_item, dim=0).tolist()
        category_list_100 = []
        for id_top_100_temp in id_top_100:
            category_temp_list = [] 
            for id_temp in id_top_100_temp:
                category_temp_list.append(id_category_dict[id_temp])
            category_list_100.append(category_temp_list)
        category_list_100.append(category_list_100)
        path_data_category = '../datasets/data/category/' + args.dataset +'/DiffuRec_top100_category.pkl'
        with open(path_data_category, 'wb') as f:
            pickle.dump(category_list_100, f)

    return best_model, test_metrics_dict_mean
    
