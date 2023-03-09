import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels, num_outs, out_channel=256):
        super(FPN, self).__init__()
        self.num_outs = num_outs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i, inc in enumerate(in_channels):
            lateral_conv = nn.Conv2d(inc, out_channel, kernel_size=(1, 1), stride=(1, 1))
            fpn_conv = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

            self.lateral_convs.append(lateral_conv)
            # self.fpn_convs.append(fpn_conv)

            if i < 3:
                self.fpn_convs.append(fpn_conv)

        # fpn_conv = nn.Conv2d(2048, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # nn.init.kaiming_normal_(fpn_conv.weight, a=0, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(fpn_conv.bias, 0)
        # self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, mode='nearest')

        # outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        # ]
        # outs.append(self.fpn_convs[-1](inputs[-1]))
        # outs.append(F.max_pool2d(outs[-1], 1, stride=2))

        outs = [
            self.fpn_convs[0](laterals[0]),
            self.fpn_convs[1](laterals[1]),
            self.fpn_convs[2](laterals[2]),
            # self.fpn_convs[3](laterals[3])
        ]

        return outs
