from dataclasses import asdict, is_dataclass

def dataclass_to_dict(obj):
    assert is_dataclass(obj) and not isinstance(obj, type), f"{type(obj).__name__} is not a dataclass instance"
    return asdict(obj)