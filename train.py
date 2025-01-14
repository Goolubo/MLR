import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import random
import copy
import matplotlib.pyplot as plt

from dataloaders.dataloader import initDataloader
from modeling.net import DRA
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
from modeling.layers import build_criterion
from torch.utils.data import DataLoader


# ========== 额外新增的工具函数 ==========

def get_all_classes(dataset_root):
    """
    获取 dataset_root 下所有子目录名作为子类名
    """
    return [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]


def aucPerformance(scores, labels, prt=True):
    """
    计算 AUC-ROC 和 AUC-PR
    """
    roc_auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap


def plot_roc_pr_curves(labels, scores, save_dir, prefix=''):
    """
    绘制并保存 ROC 曲线和 PR 曲线
    """
    # ROC
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = %.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, f'{prefix}ROC_curve.png'), dpi=300)
    plt.close()

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='blue', label='PR curve (AP = %.4f)' % ap)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(save_dir, f'{prefix}PR_curve.png'), dpi=300)
    plt.close()


# ========== 优化后的训练类 ==========

class Trainer(object):

    def __init__(self, args):
        self.args = args
        # Define Dataloader
        kwargs = {'num_workers': args.workers}
        self.train_loader, self.test_loader = initDataloader.build(args, **kwargs)

        # total_heads == 4 的情况下，需要构建 ref_loader
        if self.args.total_heads == 4:
            temp_args = copy.deepcopy(args)
            temp_args.batch_size = self.args.nRef
            temp_args.nAnomaly = 0
            self.ref_loader, _ = initDataloader.build(temp_args, **kwargs)
            self.ref = iter(self.ref_loader)

        # 初始化模型
        self.model = DRA(args, backbone=self.args.backbone)

        # 若指定了预训练权重，则加载
        if self.args.pretrain_dir is not None:
            self.model.load_state_dict(torch.load(self.args.pretrain_dir))
            print('Load pretrain weight from: ' + self.args.pretrain_dir)

        # 构建损失函数
        self.criterion = build_criterion(args.criterion)

        # 优化器 & 学习率调度
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        # 将模型和损失函数放到 GPU
        if self.args.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        # 创建日志文件
        self.log_file = os.path.join(self.args.experiment_dir, 'training_log.txt')
        with open(self.log_file, 'w') as f:
            f.write("====== Training Log ======\n")

    def generate_target(self, target, eval=False):
        """
        根据实际情况给四个头部分配不同的target
        """
        targets = list()
        if eval:
            # eval 时对应 4 个 head：0->正常，1/2/3->原始标签
            targets.append(target == 0)
            targets.append(target)
            targets.append(target)
            targets.append(target)
            return targets
        else:
            # train 时
            # target==0 表示正常样本，!=0 表示异常(其中可能区分 1,2? 视任务定义)
            temp_t = target != 0
            targets.append(target == 0)
            targets.append(temp_t[target != 2])
            targets.append(temp_t[target != 1])
            targets.append(target != 0)
            return targets

    def training(self, epoch):
        train_loss = 0.0
        class_loss = [0.0] * self.args.total_heads
        self.model.train()

        # 学习率调度
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

            losses = list()
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
            for i in range(self.args.total_heads):
                class_loss[i] += losses[i].item()

            tbar.set_description(
                f'Epoch {epoch}, Loss: {train_loss / (idx + 1):.4f}, LR: {current_lr:.6f}'
            )

        # 写入日志
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch}: Loss {train_loss / len(self.train_loader):.4f}, LR {current_lr:.6f}\n")

    def normalization(self, data):
        # 如果需要可在这里对输出进行 0-1 等归一化，这里简单返回
        return data

    def eval(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='Evaluating')
        test_loss = 0.0
        class_pred = [np.array([]) for _ in range(self.args.total_heads)]
        total_target = np.array([])

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
            tbar.set_description(f'Test Loss: {test_loss / (i + 1):.4f}')

            # 记录各 head 的输出
            for j in range(self.args.total_heads):
                if j == 0:
                    data = -1 * outputs[j].data.cpu().numpy()
                else:
                    data = outputs[j].data.cpu().numpy()
                class_pred[j] = np.append(class_pred[j], data)

            total_target = np.append(total_target, target.cpu().numpy())

        # 将多个 head 的结果做简单加和
        total_pred = self.normalization(class_pred[0])
        for i in range(1, self.args.total_heads):
            total_pred = total_pred + self.normalization(class_pred[i])

        # 写入 result.txt
        with open(os.path.join(self.args.experiment_dir, 'result.txt'), 'a+', encoding="utf-8") as w:
            for label, score in zip(total_target, total_pred):
                w.write(str(label) + '   ' + str(score) + "\n")

        # 计算 AUC
        roc_auc, ap = aucPerformance(total_pred, total_target, prt=False)
        print(f"\nEvaluation => AUC-ROC: {roc_auc:.4f}, AUC-PR: {ap:.4f}")

        # 绘制 ROC、PR 曲线
        plot_roc_pr_curves(total_target, total_pred, self.args.experiment_dir)

        # 保存柱状图
        normal_mask = (total_target == 0)
        outlier_mask = (total_target == 1)
        plt.clf()
        plt.bar(np.arange(total_pred.size)[normal_mask], total_pred[normal_mask], color='green', label='Normal')
        plt.bar(np.arange(total_pred.size)[outlier_mask], total_pred[outlier_mask], color='red', label='Outlier')
        plt.ylabel("Anomaly score")
        plt.legend()
        plt.savefig(os.path.join(self.args.experiment_dir, "score_bar.png"), dpi=300)
        plt.close()

        return roc_auc, ap

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(self.args.experiment_dir, filename))

    def load_weights(self, filename):
        self.model.load_state_dict(torch.load(os.path.join(self.args.experiment_dir, filename)))

    def init_network_weights_from_pretraining(self):
        # 若需要将自定义网络的参数加载进来，可参考此函数
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=48, help="batch size used in SGD")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="the number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=30, help="the number of epochs")
    parser.add_argument("--cont_rate", type=float, default=0.0, help="the outlier contamination rate in the training data")
    parser.add_argument("--test_threshold", type=int, default=0, help="the outlier contamination rate in the training data")
    parser.add_argument("--test_rate", type=float, default=0.0, help="the outlier contamination rate in the training data")
    parser.add_argument("--dataset", type=str, default='mvtecad', help="a list of data set names")
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the base random seed number")
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--savename', type=str, default='model.pkl', help="save modeling")
    parser.add_argument('--dataset_root', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
    parser.add_argument('--experiment_dir', type=str, default='./experiment/experiment_mlr', help="experiment directory")
    parser.add_argument('--classname', type=str, default='bottle', help="dataset class")  # 若设为 'all' 则遍历所有子类
    parser.add_argument('--img_size', type=int, default=448, help="image size")
    parser.add_argument("--nAnomaly", type=int, default=10, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='resnet18', help="backbone")
    parser.add_argument('--criterion', type=str, default='deviation', help="loss: 'CE' or 'deviation' or other custom")
    parser.add_argument("--topk", type=float, default=0.1, help="topk in MIL")
    parser.add_argument('--know_class', type=str, default=None, help="set the know class for hard setting")
    parser.add_argument('--pretrain_dir', type=str, default=None, help="root of pretrain weight")
    parser.add_argument("--total_heads", type=int, default=4, help="number of head in training")
    parser.add_argument("--nRef", type=int, default=5, help="number of reference set")
    parser.add_argument('--outlier_root', type=str, default=None, help="OOD dataset root")

    # 额外增加重复实验次数
    parser.add_argument("--repeats", type=int, default=3, help="number of repeated experiments for each sub-class")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # 若 classname 为 'all'，则遍历 dataset_root 下所有子目录，否则只针对指定子类
    if args.classname.lower() == 'all':
        sub_classes = get_all_classes(args.dataset_root)
    else:
        sub_classes = [args.classname]

    # 如果顶级实验目录不存在，则创建
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    # 保留最初的 experiment_dir 以便后续在循环中基于它来构建
    base_experiment_dir = args.experiment_dir

    # 记录所有子类的 AUC-ROC
    all_subclass_aucs = {}

    # 对每个子类进行实验
    for cls in sub_classes:
        print(f"\n================= Class: {cls} =================")
        # 为子类单独创建目录
        subclass_exp_dir = os.path.join(base_experiment_dir, cls)
        if not os.path.exists(subclass_exp_dir):
            os.makedirs(subclass_exp_dir)

        # 用于保存多次重复实验的 AUC
        subclass_roc_scores = []
        subclass_pr_scores = []

        for rep in range(args.repeats):
            print(f"\n*** Start Training [Class={cls}, Repetition={rep+1}/{args.repeats}] ***")
            # 每次重复都基于子类目录再创建 rep 目录
            rep_exp_dir = os.path.join(subclass_exp_dir, f"rep_{rep+1}")
            if not os.path.exists(rep_exp_dir):
                os.makedirs(rep_exp_dir)

            # ========== 设置随机数种子（可复现） ==========
            # 你可以自定义如何改变随机数种子，这里演示用 (基准seed + rep)
            current_seed = (args.ramdn_seed ^ ((rep + 1) * 999983)) % 2147483647
            random.seed(current_seed)
            np.random.seed(current_seed)
            torch.manual_seed(current_seed)
            if args.cuda:
                torch.cuda.manual_seed_all(current_seed)

            # 记录本次实验的随机种子到文件
            seed_log = os.path.join(rep_exp_dir, 'seed_log.txt')
            with open(seed_log, 'w') as sf:
                sf.write(f"Current random seed: {current_seed}\n")

            # 更新 args 中的子类、实验目录
            args.classname = cls
            args.experiment_dir = rep_exp_dir

            # 初始化 Trainer
            trainer = Trainer(args)

            # 将超参写到 setting.txt
            argsDict = args.__dict__
            with open(os.path.join(args.experiment_dir, 'setting.txt'), 'w') as f:
                f.write('------------------ start ------------------\n')
                for eachArg, value in argsDict.items():
                    f.write(f'{eachArg} : {value}\n')
                f.write('------------------- end -------------------\n')

            # 训练
            for epoch in range(1, args.epochs + 1):
                trainer.training(epoch)

            # 评估
            roc, pr = trainer.eval()
            subclass_roc_scores.append(roc)
            subclass_pr_scores.append(pr)

            # 保存模型
            trainer.save_weights(args.savename)

            # 记录本次重复的指标
            with open(os.path.join(args.experiment_dir, 'evaluation_log.txt'), 'a') as f:
                f.write(f"[Repetition {rep+1}] AUC-ROC: {roc:.4f}, AUC-PR: {pr:.4f}\n")

        # 统计该子类多次重复的平均和方差
        mean_roc = np.mean(subclass_roc_scores)
        std_roc = np.std(subclass_roc_scores)
        mean_pr = np.mean(subclass_pr_scores)
        std_pr = np.std(subclass_pr_scores)
        print(f"Class [{cls}] => AUC-ROC mean±std: {mean_roc:.4f} ± {std_roc:.4f}")
        print(f"Class [{cls}] => AUC-PR  mean±std: {mean_pr:.4f} ± {std_pr:.4f}")
        all_subclass_aucs[cls] = {'roc': subclass_roc_scores, 'pr': subclass_pr_scores}

        # 将统计结果写到子类目录
        with open(os.path.join(subclass_exp_dir, 'subclass_summary.txt'), 'w') as f:
            f.write(f"Class: {cls}\n")
            f.write(f"All ROC scores: {subclass_roc_scores}\n")
            f.write(f"Mean±std AUC-ROC: {mean_roc:.4f} ± {std_roc:.4f}\n")
            f.write(f"All PR scores: {subclass_pr_scores}\n")
            f.write(f"Mean±std AUC-PR: {mean_pr:.4f} ± {std_pr:.4f}\n")

    # 如果有多个子类，可以对所有子类的结果进行总体统计
    if len(sub_classes) > 1:
        flat_aucs = [v for cls_val in all_subclass_aucs.values() for v in cls_val['roc']]
        final_mean = np.mean(flat_aucs)
        final_std = np.std(flat_aucs)
        print(f"\n[Overall] => AUC-ROC mean±std: {final_mean:.4f} ± {final_std:.4f}")

        # 记录到 overall_summary
        with open(os.path.join(base_experiment_dir, 'overall_summary.txt'), 'w') as f:
            f.write(f"All Classes ROC: {flat_aucs}\n")
            f.write(f"Overall Mean±std AUC-ROC: {final_mean:.4f} ± {final_std:.4f}\n")


if __name__ == '__main__':
    main()
