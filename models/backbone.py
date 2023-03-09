# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import torch
from torch import nn
from mmdet.models.necks.fpn import FPN


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Backbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self,
                 args):
        super(Backbone, self).__init__()

        if args.backbone == 'resnet50':
            from models.resnet import resnet50
            backbone = resnet50(replace_stride_with_dilation=[False, False, False],
                                pretrained=True,
                                norm_layer=FrozenBatchNorm2d)

        elif args.backbone == 'resnext50':
            from models.resnet import resnext50_32x4d
            backbone = resnext50_32x4d(replace_stride_with_dilation=[False, False, False],
                                       pretrained=True,
                                       norm_layer=FrozenBatchNorm2d)

        elif args.backbone == 'convnext':
            from models.convnext import ConvNeXt

            backbone = ConvNeXt(depths=[3, 3, 27, 3],
                                dims=[128, 256, 512, 1024],
                                drop_path_rate=0.7,
                                layer_scale_init_value=1.0)
            url = '/data1/pdl1/pre-trained/convnext_base_1k_224.pth'
            # url = '/data1/pdl1/pre-trained/convnext_base_1k_224_ema.pth'
            backbone.init_weights(url)

        elif args.backbone == 'vitdet':
            from functools import partial
            from detectron2.config import instantiate
            from detectron2.config import LazyCall as L
            from models.vit import ViT, SimpleFeaturePyramid
            from detectron2.checkpoint import DetectionCheckpointer
            from detectron2.modeling.backbone.fpn import LastLevelMaxPool

            embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
            backbone = L(SimpleFeaturePyramid)(
                net=L(ViT)(
                    img_size=256,
                    patch_size=16,
                    embed_dim=embed_dim,
                    depth=depth,
                    num_heads=num_heads,
                    drop_path_rate=dp,
                    window_size=14,
                    mlp_ratio=4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    window_block_indexes=[
                        0,
                        1,
                        3,
                        4,
                        6,
                        7,
                        9,
                        10,
                    ],
                    residual_block_indexes=[],
                    use_rel_pos=True,
                    out_feature="last_feat",
                ),
                in_feature="${.net.out_feature}",
                out_channels=256,
                scale_factors=(4.0, 2.0, 1.0, 0.5),
                top_block=L(LastLevelMaxPool)(),
                norm="LN",
                square_pad=256,
            )

            backbone = instantiate(backbone)

            init_checkpoint = '/data1/pdl1/pre-trained/mae_pretrain_vit_base.pth'
            torch.load(init_checkpoint, map_location=torch.device("cpu"))

            checkpointer = DetectionCheckpointer(backbone)
            checkpointer.resume_or_load(init_checkpoint)

        elif args.backbone == 'swin':

            backbone = dict(
                type='SwinTransformer',
                embed_dims=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=7,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                pretrained='/data1/pdl1/pre-trained/swin_base_patch4_window7_224.pth')

            from mmdet.models.builder import build_backbone as bd
            backbone = bd(backbone)

        elif args.backbone == 'vit':
            from functools import partial
            from detectron2.config.lazy import LazyCall as L
            from MIMDet.models import MIMDetBackbone, MIMDetDecoder, MIMDetEncoder
            from detectron2.modeling.backbone.fpn import LastLevelMaxPool

            from detectron2.modeling.backbone import FPN

            backbone = L(FPN)(
                bottom_up=L(MIMDetBackbone)(
                    encoder=L(MIMDetEncoder)(
                        img_size=256,
                        patch_size=16,
                        in_chans=3,
                        embed_dim=768,
                        depth=12,
                        num_heads=12,
                        mlp_ratio=4.0,
                        dpr=0.1,
                        norm_layer=partial(nn.LayerNorm, eps=1e-5),
                        pretrained="/data1/pdl1/pre-trained/mae_pretrain_vit_base.pth",
                        checkpointing=True,
                    ),
                    decoder=L(MIMDetDecoder)(
                        img_size="${..encoder.img_size}",
                        patch_size="${..encoder.patch_size}",
                        embed_dim=768,
                        decoder_embed_dim=512,
                        depth=4,
                        num_heads=16,
                        mlp_ratio=4.0,
                        pretrained="${..encoder.pretrained}",
                        checkpointing=True,
                    ),
                    sample_ratio=0.5,
                    size_divisibility=32,
                    _out_feature_channels=[192, 384, 512, 512],
                ),
                in_features=["c2", "c3", "c4", "c5"],
                out_channels=256,
                top_block=L(LastLevelMaxPool)(),
            )

            from detectron2.config import instantiate
            backbone = instantiate(backbone)

        elif args.backbone == 'RepLKNet':
            from RepLKNet.replknet import RepLKNet
            from RepLKNet.utils import load_state_dict

            backbone = RepLKNet(large_kernel_sizes=[31, 29, 27, 13],
                                layers=[2, 2, 18, 2],
                                channels=[128, 256, 512, 1024],
                                drop_path_rate=0.6,
                                small_kernel=5,
                                dw_ratio=1,
                                num_classes=None,
                                out_indices=(0, 1, 2, 3),
                                use_checkpoint=True,
                                small_kernel_merged=False,
                                use_sync_bn=False)

            finetune = '/data1/pdl1/pre-trained/RepLKNet-31B_ImageNet-1K_224.pth'
            checkpoint = torch.load(finetune, map_location='cpu')

            model_key = 'model|module'

            checkpoint_model = None
            for model_key in model_key.split('|'):
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print("Load state_dict by model_key = %s" % model_key)
                    break
            if checkpoint_model is None:
                checkpoint_model = checkpoint

            del checkpoint_model['head.weight']
            del checkpoint_model['head.bias']

            load_state_dict(backbone, checkpoint_model, prefix='')

        elif args.backbone == 'EfficientNet':
            from mmdet.models.builder import build_backbone

            checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'  # noqa
            backbone = dict(type='EfficientNet',
                            arch='b3',
                            drop_path_rate=0.2,
                            out_indices=(2, 3, 4, 5),
                            frozen_stages=0,
                            norm_cfg=dict(type='BN', requires_grad=True),
                            norm_eval=False,
                            init_cfg=dict(
                                type='Pretrained', prefix='backbone', checkpoint=checkpoint))
            backbone = build_backbone(backbone)

        else:
            raise NotImplementedError(f'Backbone {args.backbone} is not supported yet.')

        self.backbone = backbone

        from models.fpn import FPN
        self.neck = FPN(in_channels=[256, 512, 1024, 2048],
                        out_channel=256,
                        num_outs=3)

    def forward(self, images):
        xs = self.backbone(images)

        out = self.neck(xs)
        return out


def build_backbone(args):
    backbone = Backbone(args)
    return backbone
