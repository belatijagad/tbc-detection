import torch.nn as nn


from ..backbones import build_backbone
from ..necks import build_neck
from ..heads import build_head

class SingleStageDetector(nn.Module):
    def __init__(self,
                 backbone: str,
                 neck_cfg: dict[str, any] | None,
                 bbox_head_cfg: dict[str, any]):
        super(SingleStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck_cfg.pop("name"), neck_cfg)
        self.bbox_head = build_head(bbox_head_cfg.pop("name"), bbox_head_cfg)
        
    def forward(self, x):
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        x = self.bbox_head(x)
        return x
    