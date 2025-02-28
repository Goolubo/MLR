import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import random
import copy
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.metrics import f1_score, precision_score, recall_score

# ========== 引入你项目中已有的工具函数/类 ==========
from dataloaders.dataloader import initDataloader
from modeling.net import DRA  # 或者你在 net.py 中定义的模型名称
from modeling.layers import build_criterion

import pickle

###########################################################################################
# 1、评估测试集是计算并输出AUC、F1等多种指标
# 2、可视化部分伪异常样本
# 3、渲染异常热力图
# 4、补充了重复训练和遍历所有数据类
###########################################################################################


# 全局激活缓存，用于保存 hook 输出
activation_cache = {}

def activation_hook_fn(module, input, output):
    """
    Forward hook，用于保存指定层的输出特征
    """
    global activation_cache
    activation_cache["feat"] = output.clone().detach()


def find_optimal_threshold(labels, scores):
    """
    可选：自动搜索一个最优阈值来最大化 F1或别的指标
    """
    best_thr = 0.0
    best_f1 = 0.0
    cand_thrs = np.linspace(min(scores), max(scores), 50)
    for thr in cand_thrs:
        preds = (scores >= thr).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr


def get_all_classes(dataset_root):
    """
    示例：从 dataset_root 下的文件夹名获取子类名称列表
    """
    return [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]


class Trainer(object):
    def __init__(self, args):
        self.args = args
        kwargs = {'num_workers': args.workers}

        # ========== 构建训练、测试 DataLoader ==========
        self.train_loader, self.test_loader = initDataloader.build(args, **kwargs)

        # 若 total_heads == 4，需要单独的参考图像集(ref_loader)
        if self.args.total_heads == 4:
            temp_args = copy.deepcopy(args)
            temp_args.batch_size = self.args.nRef
            temp_args.nAnomaly = 0
            self.ref_loader, _ = initDataloader.build(temp_args, **kwargs)
            self.ref = iter(self.ref_loader)

        # ========== 初始化模型 ==========
        self.model = DRA(args, backbone=self.args.backbone)

        # 如果需要加载预训练权重
        if self.args.pretrain_dir is not None:
            self.model.load_state_dict(torch.load(self.args.pretrain_dir))
            print('Load pretrain weight from: ' + self.args.pretrain_dir)

        # ========== 注册 forward hook (示例：在 resnet18 layer4[-1].conv2 等层) ==========
        # 若你的项目骨干不同，请根据实际结构适当修改
        target_layer = None
        if hasattr(self.model, "feature_extractor") and hasattr(self.model.feature_extractor, "net"):
            try:
                target_layer = self.model.feature_extractor.net.layer4[-1].conv2
                target_layer.register_forward_hook(activation_hook_fn)
                print("Forward hook registered on layer4[-1].conv2")
            except:
                print("Warning: Could not register hook. Modify target layer as needed.")

        # 构建损失函数、优化器、调度器
        self.criterion = build_criterion(args.criterion)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        if self.args.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        # 日志文件
        self.log_file = os.path.join(self.args.experiment_dir, 'training_log.txt')
        with open(self.log_file, 'w') as f:
            f.write("====== Training Log ======\n")

        # 伪异常可视化 + 激活热力图输出目录
        self.save_vis_dir = os.path.join(self.args.experiment_dir, "pseudo_vis")
        if not os.path.exists(self.save_vis_dir):
            os.makedirs(self.save_vis_dir)

    def generate_target(self, target, eval=False):
        """
        根据项目实际情况对多头网络分配不同的目标标签
        """
        # 这里只演示一个常见写法，若你原先就有可保持不变
        targets = list()
        if eval:
            targets.append(target == 0)  # normal head
            targets.append(target)
            targets.append(target)
            targets.append(target)
        else:
            temp_t = target != 0  # 是否异常
            # normal head
            targets.append(target == 0)
            # seen head
            targets.append(temp_t[target != 2])
            # pseudo head
            targets.append(temp_t[target != 1])
            # composite head
            targets.append(target != 0)
        return targets

    def combine_heads(self, outputs, target):
        """
        将多头输出组合成一个总的 anomaly score
        """
        normal_scores = outputs[0].data.cpu().numpy() * (-1.0)
        ab_scores     = outputs[1].data.cpu().numpy()
        dummy_scores  = outputs[2].data.cpu().numpy()
        comp_scores   = outputs[3].data.cpu().numpy()

        total_pred = normal_scores + ab_scores + dummy_scores + comp_scores
        return total_pred

    def visualize_pseudo_sample(self, image_tensor, epoch, batch_idx, sample_idx):
        """
        将某张伪异常图像保存到本地 (self.save_vis_dir)
        """
        import matplotlib.pyplot as plt
        img_np = image_tensor.cpu().numpy()
        img_np = np.transpose(img_np, (1,2,0))  # (H,W,C)
        # 反归一化操作(若需要)，此处仅简单截断 0~1
        img_np = np.clip(img_np, 0, 1)

        plt.figure()
        plt.imshow(img_np)
        plt.title("Pseudo anomaly sample")
        save_path = os.path.join(
            self.save_vis_dir, f"epoch{epoch}_batch{batch_idx}_sample{sample_idx}_pseudo.png"
        )
        plt.savefig(save_path)
        plt.close()

    def visualize_activation_heatmap(self, image_tensor, epoch, batch_idx, sample_idx):
        """
        从 activation_cache["feat"] 中取特征图并可视化
        """
        import matplotlib.pyplot as plt
        import cv2
        feat = activation_cache.get("feat", None)
        if feat is None:
            return

        feat_map = feat[sample_idx]
        feat_map_mean = torch.mean(feat_map, dim=0).cpu().numpy()
        feat_map_mean = (feat_map_mean - feat_map_mean.min()) / (feat_map_mean.max() - feat_map_mean.min() + 1e-6)

        c,h,w = image_tensor.shape
        feat_map_resize = cv2.resize(feat_map_mean, (w,h))

        img_np = image_tensor.cpu().numpy()
        img_np = np.transpose(img_np, (1,2,0))
        img_np = np.clip(img_np, 0, 1)

        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.imshow(img_np)
        plt.title("Pseudo anomaly")

        plt.subplot(1,2,2)
        plt.imshow(feat_map_resize, cmap='jet')
        plt.title("Activation heatmap")
        plt.colorbar()
        save_path = os.path.join(self.save_vis_dir,
                        f"epoch{epoch}_batch{batch_idx}_sample{sample_idx}_heatmap.png")
        plt.savefig(save_path)
        plt.close()


    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']

        tbar = tqdm(self.train_loader, desc=f'Epoch {epoch} Training')
        for idx, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            if self.args.total_heads == 4:
                try:
                    ref_image = next(self.ref)['image']
                except StopIteration:
                    self.ref = iter(self.ref_loader)
                    ref_image = next(self.ref)['image']
                ref_image = ref_image.cuda()
                image = torch.cat([ref_image, image], dim=0)

            outputs = self.model(image, target)
            targets = self.generate_target(target)

            losses = []
            for i in range(self.args.total_heads):
                if self.args.criterion == 'CE':
                    prob = F.softmax(outputs[i], dim=1)
                    losses.append(self.criterion(prob, targets[i].long()).view(-1, 1))
                else:
                    losses.append(self.criterion(outputs[i], targets[i].float()).view(-1, 1))

            loss = torch.cat(losses)
            loss = torch.sum(loss)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            train_loss += loss.item()

            offset = self.args.nRef if self.args.total_heads == 4 else 0
            pseudo_idx = (target == 2).nonzero(as_tuple=True)[0]
            for pid in pseudo_idx[:2]:
                real_pid = pid + offset
                if real_pid < image.shape[0]:
                    self.visualize_pseudo_sample(image[real_pid], epoch, idx, pid.item())
                    self.visualize_activation_heatmap(image[real_pid], epoch, idx, pid.item())

            tbar.set_description(
                f'Epoch {epoch}, Loss: {train_loss/(idx+1):.4f}, LR: {current_lr:.6f}'
            )

        # 写训练日志
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch}: Loss {train_loss / len(tbar):.4f}, LR {current_lr:.6f}\n")

    def eval(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='Evaluating')
        test_loss = 0.0
        all_scores = []
        all_labels = []

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            if self.args.total_heads == 4:
                try:
                    ref_image = next(self.ref)['image']
                except StopIteration:
                    self.ref = iter(self.ref_loader)
                    ref_image = next(self.ref)['image']
                ref_image = ref_image.cuda()
                image = torch.cat([ref_image, image], dim=0)

            with torch.no_grad():
                outputs = self.model(image, target)
                targets = self.generate_target(target, eval=True)
                losses = []
                for j in range(self.args.total_heads):
                    if self.args.criterion == 'CE':
                        prob = F.softmax(outputs[j], dim=1)
                        losses.append(self.criterion(prob, targets[j].long()))
                    else:
                        losses.append(self.criterion(outputs[j], targets[j].float()))
            loss = sum(losses)
            test_loss += loss.item()

            total_pred = self.combine_heads(outputs, target)
            all_scores.extend(total_pred)
            all_labels.extend(target.cpu().numpy())

            tbar.set_description(f'Test Loss: {test_loss/(i+1):.4f}')

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        roc_auc = roc_auc_score(all_labels, all_scores)

        best_thr = find_optimal_threshold(all_labels, all_scores)
        preds_bin = (all_scores >= best_thr).astype(int)
        f1 = f1_score(all_labels, preds_bin)
        precision = precision_score(all_labels, preds_bin)
        recall = recall_score(all_labels, preds_bin)

        print(f"\nEval => AUC: {roc_auc:.4f}, F1: {f1:.4f}, P: {precision:.4f}, R: {recall:.4f}, thr={best_thr:.3f}")

        eval_log_path = os.path.join(self.args.experiment_dir, 'evaluation_log.txt')
        with open(eval_log_path, 'a') as f:
            f.write(f"AUC: {roc_auc:.4f}, F1: {f1:.4f}, P: {precision:.4f}, R: {recall:.4f}, thr={best_thr:.3f}\n")

        return roc_auc, f1

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(self.args.experiment_dir, filename))

    def load_weights(self, filename):
        self.model.load_state_dict(torch.load(os.path.join(self.args.experiment_dir, filename)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=48, help="batch size used in SGD")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="the number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=30, help="the number of epochs")
    parser.add_argument("--cont_rate", type=float, default=0.0, help="the outlier contamination rate in the training data")
    parser.add_argument("--test_threshold", type=int, default=0, help="the outlier contamination rate in the training data")
    parser.add_argument("--test_rate", type=float, default=0.0, help="the outlier contamination rate in the training data")
    parser.add_argument("--dataset", type=str, default='mvtecad', help="dataset name")
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the base random seed number")
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--savename', type=str, default='model.pkl', help="save modeling")
    parser.add_argument('--dataset_root', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
    parser.add_argument('--experiment_dir', type=str, default='./experiment/experiment_mlr', help="experiment directory")
    parser.add_argument('--classname', type=str, default='transistor', help="dataset class")
    parser.add_argument('--img_size', type=int, default=448, help="image size")
    parser.add_argument("--nAnomaly", type=int, default=10, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales")
    parser.add_argument('--backbone', type=str, default='resnet18', help="backbone name")
    parser.add_argument('--criterion', type=str, default='deviation', help="loss type: 'CE' / 'deviation' / ...")
    parser.add_argument("--topk", type=float, default=0.1, help="topk in MIL")
    parser.add_argument('--know_class', type=str, default=None, help="which class to know in training")
    parser.add_argument('--pretrain_dir', type=str, default=None, help="pretrain model path")
    parser.add_argument("--total_heads", type=int, default=4, help="number of heads in training")
    parser.add_argument("--nRef", type=int, default=5, help="number of reference images")
    parser.add_argument('--outlier_root', type=str, default=None, help="OOD dataset root")
    parser.add_argument("--repeats", type=int, default=3, help="number of repeated experiments")

    # parser.add_argument("--num_clusters", type=int, default=5, help="K-Means 聚类的类别数")
    # parser.add_argument("--topk", type=int, default=3, help="KNN 选择的最近邻数")
    # parser.add_argument("--distance_metric", type=str, default="euclidean",
    #                     choices=["euclidean", "cosine"], help="计算样本相似度的方法")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # ========== 如果 classname 为 'all'，则遍历 dataset_root 下所有子目录，否则只针对指定子类 ==========
    if args.classname.lower() == 'all':
        sub_classes = get_all_classes(args.dataset_root)
    else:
        sub_classes = [args.classname]

    # 保留最初的 experiment_dir
    base_experiment_dir = args.experiment_dir

    for cls in sub_classes:
        print(f"\n================= Class: {cls} =================")
        # 针对每个子类建一个目录
        subclass_exp_dir = os.path.join(base_experiment_dir, cls)
        if not os.path.exists(subclass_exp_dir):
            os.makedirs(subclass_exp_dir)

        # 重复多次训练
        for rep in range(1, args.repeats + 1):
            print(f"----- Repetition: {rep}/{args.repeats} -----")
            rep_exp_dir = os.path.join(subclass_exp_dir, f"rep_{rep}")
            os.makedirs(rep_exp_dir, exist_ok=True)

            # 设置随机数种子
            current_seed = args.ramdn_seed + rep * 100
            args.ramdn_seed = current_seed
            random.seed(current_seed)
            np.random.seed(current_seed)
            torch.manual_seed(current_seed)
            if args.cuda:
                torch.cuda.manual_seed_all(current_seed)

            # 更新 args
            args.classname = cls
            args.experiment_dir = rep_exp_dir

            # 初始化 Trainer
            trainer = Trainer(args)

            # 记录超参
            with open(os.path.join(args.experiment_dir, 'setting.txt'), 'w') as f:
                f.write('------------------ start ------------------\n')
                for eachArg, value in args.__dict__.items():
                    f.write(f'{eachArg} : {value}\n')
                f.write('------------------- end -------------------\n')

            # 训练
            for epoch in range(1, args.epochs + 1):
                trainer.training(epoch)
                # 每5轮或最后1轮评估
                if epoch % 5 == 0 or epoch == args.epochs:
                    roc, f1 = trainer.eval()
                    trainer.save_weights(args.savename)

    print("All repeated experiments finished.")


if __name__ == '__main__':
    main()
