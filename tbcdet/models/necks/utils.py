from .fpn import FPN


def build_neck(name: str, cfg: dict[str, any]):
    neck = None
    match name:
        case "fpn":
            neck = FPN(**cfg)
        case _:
            neck = None
    return neck
