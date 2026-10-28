import torch.nn as nn

from .base import AnchorHead
from ...blocks import ConvModule


class RetinaHead(AnchorHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 feat_channels: int,
                 loss_cls,
                 loss_bbox,
                 stacked_convs: int = 4):
        self.stacked_convs = stacked_convs
        
        super(AnchorHead, self).__init__(
            num_classes,
            in_channels,
            feat_channels,
            loss_cls,
            loss_bbox
        )
        
    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_conv = nn.ModuleList()
        self.reg_conv = nn.ModuleList()
        
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_conv.append(ConvModule(chn, self.feat_channels, 3, 1, 1))
            self.reg_conv.append(ConvModule(chn, self.feat_channels, 3, 1, 1))
            
        self.retina_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.num_classes, 3, padding=1)
        self.retina_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 3, padding=1)
        
    def forward(self, x):
        for cls_conv in self.cls_conv:
            x_cls = cls_conv(x)
        for reg_conv in self.reg_conv:
            x_reg = reg_conv(x)
            
        cls_score = self.retina_cls(x_cls)
        bbox_pred = self.retina_reg(x_reg)
        
        return cls_score, bbox_pred
