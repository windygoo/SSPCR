# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import copy

import numpy as np
import torch.nn.functional as F

from torch import nn
from models.backbone import build_backbone


class AnchorPoints(nn.Module):
    def __init__(self, row=2, col=2, grid_scale=(32, 32)):
        super(AnchorPoints, self).__init__()
        x_space = grid_scale[0] / row
        y_space = grid_scale[1] / col
        self.deltas = np.array(
            [
                [-x_space, -y_space],
                [x_space, -y_space],
                [0, 0],
                [-x_space, y_space],
                [x_space, y_space]
            ]
        ) / 2

        self.grid_scale = np.array(grid_scale)

    def forward(self, images):
        bs, _, h, w = images.shape
        centers = np.stack(
            np.meshgrid(
                np.arange(np.ceil(w / self.grid_scale[0])) + 0.5,
                np.arange(np.ceil(h / self.grid_scale[1])) + 0.5),
            -1) * self.grid_scale

        anchors = np.expand_dims(centers, 2) + self.deltas
        anchors = torch.from_numpy(anchors).float().to(images.device)

        return anchors.flatten(0, 2).repeat(bs, 1, 1)

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbone, hidden_dim, num_classes, row, col):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            num_classes: number of object classes
        """
        super().__init__()
        self.backbone = backbone
        self.get_aps = AnchorPoints(row, col)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_levels = self.backbone.neck.num_outs
        self.strides = [2 ** (i + 1) for i in range(self.num_levels)]

        # deformable roi pooling
        self.deformable_mlps = _get_clones(MLP(hidden_dim, hidden_dim, 2, 2), self.num_levels)
        for mlp in self.deformable_mlps:
            nn.init.constant_(mlp.layers[-1].bias.data, 0)
            nn.init.constant_(mlp.layers[-1].weight.data, 0)

        self.reg_head = MLP(hidden_dim, hidden_dim, 2, 3)
        self.cls_head = nn.Linear(hidden_dim, num_classes + 1)

        self.loc_aggr = nn.Sequential(nn.Linear(hidden_dim * self.num_levels, hidden_dim), nn.ReLU(inplace=True),
                                      nn.Linear(hidden_dim, self.num_levels))
        self.cls_aggr = nn.Sequential(nn.Linear(hidden_dim * self.num_levels, hidden_dim), nn.ReLU(inplace=True),
                                      nn.Linear(hidden_dim, self.num_levels))

    def forward(self, images):
        anchors = self.get_aps(images)
        features = self.backbone(images)

        reg_features, cls_features = self.extract_features(features, anchors)
        pnt_coords = self.reg_head(reg_features) + anchors
        cls_logits = self.cls_head(cls_features)

        outputs = {'pnt_coords': pnt_coords, 'cls_logits': cls_logits}
        return outputs

    def extract_features(self, features, points, align_corners=True):
        roi_features = torch.zeros(self.num_levels, *points.shape[:2], self.hidden_dim).cuda(points.device)
        for i, stride in enumerate(self.strides):
            h, w = features[i].shape[2:]
            scale = torch.tensor([w, h], dtype=torch.float, device=points.device)
            grid = (2.0 * points / stride / scale - 1.0).unsqueeze(2)  # for alignment

            pre_roi_features = F.grid_sample(features[i], grid, align_corners=align_corners).squeeze(-1).permute(0, 2,
                                                                                                                 1)
            grid = (2.0 * (points + self.deformable_mlps[i](pre_roi_features)) / stride / scale - 1.0).unsqueeze(2)

            roi_features[i] = F.grid_sample(features[i], grid, align_corners=align_corners).squeeze(-1).permute(0, 2, 1)

        roi_features = roi_features.permute(1, 2, 0, 3)
        attn_features = roi_features.flatten(2)

        reg_attn = F.softmax(self.loc_aggr(attn_features), dim=-1).unsqueeze(-1)
        reg_features = (reg_attn * roi_features).sum(dim=2)

        cls_attn = F.softmax(self.cls_aggr(attn_features), dim=-1).unsqueeze(-1)
        cls_features = (cls_attn * roi_features).sum(dim=2)

        return reg_features, cls_features


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def build_model(args):
    backbone = build_backbone(args)

    model = DETR(
        backbone,
        row=args.row,
        col=args.col,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
    )

    return model
