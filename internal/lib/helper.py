import hashlib
import importlib
from datetime import datetime
from typing import Any


def dynamic_import(module_name: str, symbol_name: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def add_attribute(attr_name: str, attr_value: Any):
    def decorator(func):
        setattr(func, attr_name, attr_value)
        return func

    return decorator


def generate_text_hash(text: str) -> str:
    text = str(text) + "None"

    return hashlib.sha3_256(text.encode()).hexdigest()


def datetime_to_timestamp(dt: datetime) -> int:
    if dt is None:
        return 0
    return int(dt.timestamp())
