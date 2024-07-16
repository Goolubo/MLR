import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.networks.backbone import build_feature_extractor, NET_OUT_DIM


class HolisticHead(nn.Module):
    def __init__(self, in_dim, dropout=0):
        super(HolisticHead, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return torch.abs(x)
class HolisticHeadwr(nn.Module):
    def __init__(self, in_dim, dropout=0):
        super(HolisticHeadwr, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dim, in_dim, 3, padding=1),
                                  nn.BatchNorm2d(in_dim),
                                  nn.ReLU(),

                                  )
        self.fc1 = nn.Linear(in_dim*2, 256)
        self.fc2 = nn.Linear(256, 1)
        self.drop = nn.Dropout(dropout)


    def forward(self, x, xr):
        xr=self.conv(xr)
        x=torch.cat((x,xr),1)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return torch.abs(x)

class HolisticHeadwr_0(nn.Module):
    def __init__(self, in_dim, dropout=0):
        super(HolisticHeadwr_0, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dim, in_dim, 3, padding=1),
                                  nn.BatchNorm2d(in_dim),
                                  nn.ReLU(),

                                  )
        self.fc1 = nn.Linear(in_dim*2, 256)
        self.fc2 = nn.Linear(256, 1)
        self.drop = nn.Dropout(dropout)


    def forward(self, x, xr):
        xr=torch.zeros(xr.size()).cuda()#self.conv(xr)
        x=torch.cat((x,xr),1)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return torch.abs(x)

class PlainHead(nn.Module):
    def __init__(self, in_dim, topk_rate=0.1):
        super(PlainHead, self).__init__()
        self.scoring = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1, padding=0)
        self.topk_rate = topk_rate

    def forward(self, x):
        x = self.scoring(x)
        x = x.view(int(x.size(0)), -1)
        topk = max(int(x.size(1) * self.topk_rate), 1)
        x = torch.topk(torch.abs(x), topk, dim=1)[0]
        x = torch.mean(x, dim=1).view(-1, 1)
        return x


class CompositeHead(PlainHead):
    def __init__(self, in_dim, topk=0.1):
        super(CompositeHead, self).__init__(in_dim, topk)
        self.conv = nn.Sequential(nn.Conv2d(in_dim, in_dim, 3, padding=1),
                                  nn.BatchNorm2d(in_dim),
                                  nn.ReLU())

    def forward(self, x, ref):
        ref = torch.mean(ref, dim=0).repeat([x.size(0), 1, 1, 1])
        x = ref - x
        x = self.conv(x)
        x = super().forward(x)
        return x
        

class CompositeHeadwr_0(PlainHead):
    def __init__(self, in_dim, topk=0.1):
        super(CompositeHeadwr_0, self).__init__(in_dim, topk)
        self.conv = nn.Sequential(nn.Conv2d(in_dim, in_dim, 3, padding=1),
                                  nn.BatchNorm2d(in_dim),
                                  nn.ReLU())

    def forward(self, x, ref):
        ref = torch.mean(ref, dim=0).repeat([x.size(0), 1, 1, 1])
        x = ref - x
        xr = self.conv(x)
        x = super().forward(xr)
        return x,xr

class CompositeHeadwr(PlainHead):
    def __init__(self, in_dim, topk=0.1):
        super(CompositeHeadwr, self).__init__(in_dim, topk)
        self.conv = nn.Sequential(nn.Conv2d(in_dim, in_dim, 3, padding=1),
                                  nn.BatchNorm2d(in_dim),
                                  nn.ReLU(),

                                  )

    def forward(self, x, ref):
        #ckidx1=ref.size(0)//3-1
        #ckidx2=ref.size(0)*2//3-1
        #ckidx3=ref.size(0)-1
        
        for i in range(ref.size(0)):
            ref0 = ref[i, :, :, :].repeat([x.size(0), 1, 1, 1])
            x0 = ref0 - x
            x0 = self.conv(x0)
            if i==0:
                xr=x0
                ckx0 = super().forward(x0)
            else:
                xr=xr+x0
                ckx0 += super().forward(xr/(i+1))
            
            
            '''
            if i==ckidx1:
                xr1=xr/(1+i)
                ckx1 = super().forward(xr1)
            if i==ckidx2:
                xr2=xr/(1+i)
                ckx2 = super().forward(xr2)
            if i==ckidx3:
                xr3=xr/(1+i)
                ckx3 = super().forward(xr3)
            '''
      
        xr=xr/ref.size(0)

        #combined_tensor = torch.cat((x0.unsqueeze(0), x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0),x4.unsqueeze(0)),0)
        #xr, _ = torch.min(combined_tensor, dim=0)

        x = ckx0/ref.size(0)
        return x,xr
class DRA(nn.Module):
    def __init__(self, cfg, backbone="resnet18"):
        super(DRA, self).__init__()
        self.cfg = cfg
        self.feature_extractor = build_feature_extractor(backbone, cfg)
        self.in_c = NET_OUT_DIM[backbone]
        self.holistic_head = HolisticHeadwr_0(self.in_c)
        self.seen_head = PlainHead(self.in_c, self.cfg.topk)
        self.pseudo_head = PlainHead(self.in_c, self.cfg.topk)
        self.composite_head = CompositeHeadwr(self.in_c, self.cfg.topk)

    def forward(self, image, label):
        image_pyramid = list()
        for i in range(self.cfg.total_heads):
            image_pyramid.append(list())
        for s in range(self.cfg.n_scales):
            image_scaled = F.interpolate(image, size=self.cfg.img_size // (2 ** s)) if s > 0 else image
            feature = self.feature_extractor(image_scaled)

            ref_feature = feature[:self.cfg.nRef, :, :, :]
            feature = feature[self.cfg.nRef:, :, :, :]

            if self.training:
                comparison_scores,xr = self.composite_head(feature, ref_feature)
                normal_scores = self.holistic_head(feature, xr)
                abnormal_scores = self.seen_head(feature[label != 2])
                dummy_scores = self.pseudo_head(feature[label != 1])

            else:
                comparison_scores,xr = self.composite_head(feature, ref_feature)
                normal_scores = self.holistic_head(feature, xr)
                abnormal_scores = self.seen_head(feature)
                dummy_scores = self.pseudo_head(feature)

            for i, scores in enumerate([normal_scores, abnormal_scores, dummy_scores, comparison_scores]):
                image_pyramid[i].append(scores)
        for i in range(self.cfg.total_heads):
            image_pyramid[i] = torch.cat(image_pyramid[i], dim=1)
            image_pyramid[i] = torch.mean(image_pyramid[i], dim=1)
        return image_pyramid


