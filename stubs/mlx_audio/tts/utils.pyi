from pathlib import Path
from typing import Any

def load(model_path: str | Path, lazy: bool = False, strict: bool = True, **kwargs: Any) -> Any: ...
