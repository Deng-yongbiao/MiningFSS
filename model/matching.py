import model.resnet as resnet

import torch
from torch import nn
import torch.nn.functional as F


class MatchingNet(nn.Module):
    def __init__(self, backbone):
        super(MatchingNet, self).__init__()
        self.backbone = resnet.__dict__[backbone](pretrained=True)

    def forward(self, img_s_list, mask_s_list, img_q):
        h, w = img_q.shape[-2:]
        output_shape = (h, w)
        # feature maps of support images
        feature_s_list = []
        for k in range(len(img_s_list)):
            feature_s_list.append(self.backbone.base_forward(img_s_list[k])["feat4"])
        # feature map of query image
        feature_q_all = self.backbone.base_forward(img_q)
        feature_q = feature_q_all["feat4"]
        # print("feature_q shape:", feature_q.shape)

        # foreground(target class) and background prototypes pooled from K support features
        feature_fg_list = []
        feature_bg_list = []
        for k in range(len(img_s_list)):
            feature_fg_list.append(self.masked_average_pooling(feature_s_list[k],
                                                               (mask_s_list[k] == 1).float())[None, :])
            feature_bg_list.append(self.masked_average_pooling(feature_s_list[k],
                                                               (mask_s_list[k] == 0).float())[None, :])
        # average K foreground prototypes and K background prototypes
        feature_fg = torch.mean(torch.cat(feature_fg_list, dim=0), dim=0)
        feature_bg = torch.mean(torch.cat(feature_bg_list, dim=0), dim=0) 
        # print("feature_fg shape:", feature_fg.shape)
        # print("feature_bg shape:", feature_bg.shape)


        # measure the similarity of query features to fg/bg prototypes
        similarity_fg = F.cosine_similarity(feature_q, feature_fg[..., None, None], dim=1)
        similarity_bg = F.cosine_similarity(feature_q, feature_bg[..., None, None], dim=1)

        # print("similarity_fg shape:", similarity_fg.shape)
        # print("similarity_bg shape:", similarity_bg.shape)

        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        # print("out shape:", out.shape)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        # print("out shape:", out.shape)
        # print("=====================================")

        auxiliary_feat = self.get_auxiliary_feature_map(feature_q_all["feat4"], output_shape)

        return out, feature_q_all, similarity_fg, similarity_bg, auxiliary_feat

    def masked_average_pooling(self, feature, mask):
        feature = F.interpolate(feature, size=mask.shape[-2:], mode="bilinear", align_corners=True)
        masked_feature = torch.sum(feature * mask[:, None, ...], dim=(2, 3)) \
                         / (mask[:, None, ...].sum(dim=(2, 3)) + 1e-5)
        return masked_feature
    def get_auxiliary_feature_map(self, feats, output_shape):
        channel_mean = torch.mean(feats,dim=1,keepdim=True)
        channel_mean = F.interpolate(channel_mean, size=output_shape, mode='bilinear', align_corners=False)
        channel_mean = channel_mean.squeeze(1)
        # channel_mean = (((channel_mean - torch.min(channel_mean))/(torch.max(channel_mean)-torch.min(channel_mean)))) #归一化
        return channel_mean