from torchvision.models import (
    vgg19, VGG19_BN_Weights,
    resnet50, ResNet50_Weights,
    resnext50_32x4d, ResNeXt50_32X4D_Weights,
)


def get_backbone(backbone: str):
    match backbone:
        case "vggnet":
            backbone = vgg19(weights=VGG19_BN_Weights.DEFAULT)
        case "resnet":
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        case "resnext":
            backbone = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        case _:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
    return backbone
