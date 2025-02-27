# 文件: tools/build_normal_feats_per_class.py
import os
import sys
import torch
import numpy as np
import argparse

# 假设需要用到项目中的一些自定义模块：
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.net import DRA
from dataloaders.dataloader import initDataloader
from dataloaders.utlis import worker_init_fn_seed

from torchvision import transforms
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser("Build Normal Feats for each sub-class without FAISS")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="mvtecad")
    parser.add_argument("--dataset_root", type=str, default="D:\Study\pycharm_project\MLR\data\mvtec_anomaly_detection")
    parser.add_argument("--experiment_dir", type=str, default="D:\Study\pycharm_project\MLR\experiment", help="where to save feats")
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--nAnomaly", type=int, default=10)
    parser.add_argument("--cont_rate", type=float, default=0.0)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--total_heads", type=int, default=4)
    parser.add_argument("--nRef", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ramdn_seed", type=int, default=42)
    parser.add_argument("--topk", type=float, default=0.1, help="topk in MIL")
    parser.add_argument('--know_class', type=str, default=None, help="which class to know in training")
    parser.add_argument("--test_threshold", type=int, default=0,
                        help="the outlier contamination rate in the training data")
    parser.add_argument("--test_rate", type=float, default=0.0,
                        help="the outlier contamination rate in the training data")
    parser.add_argument('--outlier_root', type=str, default=None, help="OOD dataset root")
    # 假设我们有多个子类:
    parser.add_argument("--sub_classes", nargs="+", default=[
        "carpet","grid","leather","tile","wood",  # ... etc
    ])
    return parser.parse_args()

def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def main():
    args = parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    # 初始化一个模型
    model = DRA(args, backbone=args.backbone)
    if args.cuda:
        model.cuda()
    model.eval()

    transform_fn = build_transform(args.img_size)

    for cls_name in args.sub_classes:
        print(f"\n===== Building Normal Features for subclass: {cls_name} =====")
        # 1) 指定子类: 让 dataloader 只载入该子类
        args.classname = cls_name
        train_loader, _ = initDataloader.build(args)

        all_feats = []
        all_paths = []

        with torch.no_grad():
            for i in range(len(train_loader.dataset)):
                sample = train_loader.dataset[i]  # 这里根据你的 dataset 实际实现
                label = sample["label"]
                path  = sample["imgpath"] if "imgpath" in sample else None
                if label != 0:
                    continue  # 只收集正常

                if "origin" in sample:
                    img_pil = sample["origin"]
                else:
                    # 如果 dataset 没有保存原图，需要自己根据 path 打开
                    img_pil = Image.open(os.path.join(args.dataset_root, path)).convert("RGB")

                img_tensor = transform_fn(img_pil).unsqueeze(0)  # =>(1,C,H,W)
                if args.cuda:
                    img_tensor = img_tensor.cuda()

                feat = model.feature_extractor(img_tensor)  # =>(1,512,h',w')
                feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1,1))  # =>(1,512,1,1)
                feat = feat.view(1,-1)  # =>(1,512)
                feat_np = feat.cpu().numpy().astype(np.float32)[0]

                all_feats.append(feat_np)
                all_paths.append(path if path else f"{cls_name}_normal_{i}.png")

        all_feats = np.stack(all_feats, axis=0)  # shape: (N,512)
        print(f"{cls_name}: collected normal feats = {all_feats.shape[0]}")

        if all_feats.shape[0] < 1:
            print(f"[Warning] Subclass {cls_name} no normal data found, skip.")
            continue

        # 保存
        save_dir = os.path.join(args.experiment_dir, "normal_feats", cls_name)
        os.makedirs(save_dir, exist_ok=True)

        feats_path = os.path.join(save_dir, f"{cls_name}_normal_feats.npy")
        np.save(feats_path, {"feats": all_feats, "paths": all_paths})
        print(f"Saved normal feats => {feats_path}")

if __name__ == "__main__":
    main()
