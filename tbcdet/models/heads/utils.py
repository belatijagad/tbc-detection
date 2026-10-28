from .dense import RetinaHead


def build_head(name: str, cfg: dict[str, any]):
    head = None
    match name:
        case "retina":
            head = RetinaHead(**cfg)
        case _:
            head = None
    return head
