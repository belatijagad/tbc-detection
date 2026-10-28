from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import ConvModule


class FPN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_outs: int,
                 start_level: int = 0,
                 end_level: int = -1,
                 add_extra_convs: Literal["on_input", "on_lateral", "on_output", None] = None):
        super(FPN, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        
        self.start_level = start_level
        self.end_level = end_level
        self.backbone_end_level = self.num_ins if end_level == -1 else end_level
        
        self.add_extra_convs = add_extra_convs
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, 1) # 1x1 conv
            fpn_conv = ConvModule(out_channels, out_channels, 3, padding=1) # conv to predict
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = self.in_channels[self.backbone_end_level - 1] if i == 0 \
                    and self.add_extra_convs == "on_input" else out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                )
                self.fpn_convs.append(extra_fpn_conv)
        
    def forward(self, x):
        laterals = [
            lateral_conv(x[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2)
            
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                match self.add_extra_convs:
                    case "on_input":
                        extra_source = x[self.backbone_end_level - 1]
                    case "on_lateral":
                        extra_source = laterals[-1]
                    case "on_output":
                        extra_source = outs[-1]
                    case _:
                        raise NotImplementedError
                    
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    outs.append(self.fpn_convs[i](outs[-1]))
                
        return tuple(outs)
    