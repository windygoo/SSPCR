# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from mmdet.core import multi_apply
from models.backbone import build_backbone


class AnchorPoints(nn.Module):
    def __init__(self, space=16):
        super(AnchorPoints, self).__init__()
        self.space = space

    def forward(self, images):
        bs, _, h, w = images.shape
        anchors = np.stack(
            np.meshgrid(
                np.arange(np.ceil(w / self.space)),
                np.arange(np.ceil(h / self.space))),
            -1) * self.space

        origin_coord = np.array([w % self.space or self.space, h % self.space or self.space]) / 2
        anchors += origin_coord

        anchors = torch.from_numpy(anchors).float().to(images.device)
        return anchors.flatten(0, 1).repeat(bs, 1, 1)


class DPAP2PNet(nn.Module):
    """ This is the Proposal-aware P2PNet module that performs cell recognition """

    def __init__(self,
                 backbone,
                 num_classes,
                 dropout=0.1,
                 space: int = 16,
                 hidden_dim: int = 256):
        """
            Initializes the model.
        """
        super().__init__()
        self.backbone = backbone
        self.get_aps = AnchorPoints(space)
        self.hidden_dim = hidden_dim
        self.num_levels = self.backbone.neck.num_outs
        self.strides = [2 ** (i + 2) for i in range(self.num_levels)]

        self.deform_layer = MLP(hidden_dim, hidden_dim, 2, 2, dp=dropout)

        self.reg_head = MLP(hidden_dim * self.num_levels, hidden_dim, 2, 2, dp=dropout)
        self.cls_head = MLP(hidden_dim * self.num_levels, hidden_dim, 2, num_classes + 1, dp=dropout)

    def forward(self,
                images):
        bs, c, h, w = images.shape
        feats, proposals = self.backbone(images), self.get_aps(images)
        feat_sizes = [torch.tensor(feat.shape[:1:-1], dtype=torch.float, device=proposals.device) for feat in feats]

        # deformable proposal points
        roi_features = self.exf_single_layer(feats[0],
                                             proposals,
                                             feat_sizes[0],
                                             self.strides[0])
        deformable_proposals = proposals + self.deform_layer(roi_features)

        roi_features = multi_apply(self.exf_single_layer,
                                   feats,
                                   [deformable_proposals] * self.num_levels,
                                   feat_sizes,
                                   self.strides)

        roi_features = torch.stack([torch.stack(roi_features[i]) for i in range(bs)])
        roi_features = roi_features.transpose(1, 2)

        reg_features = roi_features.flatten(2)
        cls_features = roi_features.flatten(2)

        pred_coords = deformable_proposals + self.reg_head(reg_features)
        pred_logits = self.cls_head(cls_features)

        out = {'pred_coords': pred_coords, 'pred_logits': pred_logits}
        return out

    @staticmethod
    def exf_single_layer(feat: torch.Tensor,
                         points: torch.Tensor,
                         feat_size: torch.Tensor,
                         stride: float):
        grid = (2.0 * points / stride / feat_size - 1.0).unsqueeze(2)
        roi_features = F.grid_sample(feat, grid, align_corners=True).squeeze(-1).permute(0, 2, 1)
        return roi_features


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dp=0.1):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList()

        for n, k in zip([input_dim] + h, h):
            self.layers.append(nn.Linear(n, k))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(dp))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


def build_model(args):
    backbone = build_backbone(args)

    model = DPAP2PNet(
        backbone,
        args.num_classes,
        dropout=args.dropout,
        space=args.space,
        hidden_dim=args.hidden_dim,
    )
    return model
