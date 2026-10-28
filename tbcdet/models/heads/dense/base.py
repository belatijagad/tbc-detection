import torch.nn as nn


class AnchorHead(nn.Module):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 feat_channels: int,
                 loss_cls: dict,
                 loss_bbox: dict):
        super(AnchorHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        
        self.num_anchors = 4
        
        self._init_layers()
        
    def _init_layers(self):
        self.conv_cls = nn.Conv2d(self.in_channels, self.num_anchors * self.num_classes, 1)
        self.conv_bbox = nn.Conv2d(self.in_channels, self.num_anchors * 4, 1)
        
    def forward(self, x):
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_bbox(x)
        return cls_score, bbox_pred
        