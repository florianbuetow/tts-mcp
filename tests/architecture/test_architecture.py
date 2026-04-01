"""Architecture import rule tests for mistral-text-to-spech.

Enforces module boundaries:
- src/server.py must not import from src/main (no CLI in server)
- src/main.py must not import from src/server (no server in CLI)
- Both may import from src/tts (shared engine)
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pytestarch import EvaluableArchitecture

pytestmark = pytest.mark.architecture

_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"


def _get_imports(filepath: Path) -> set[str]:
    """Extract all import module names from a Python file."""
    tree = ast.parse(filepath.read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


def test_evaluable_is_configured(evaluable: EvaluableArchitecture) -> None:
    """Verify the evaluable architecture graph was built successfully."""
    assert evaluable is not None


def test_server_does_not_import_main() -> None:
    """Server module must not depend on CLI module."""
    server_file = _SRC_DIR / "server.py"
    if not server_file.exists():
        pytest.skip("src/server.py not yet created")
    imports = _get_imports(server_file)
    assert "src.main" not in imports, "src/server.py must not import from src.main"


def test_main_does_not_import_server() -> None:
    """CLI module must not depend on server module."""
    main_file = _SRC_DIR / "main.py"
    imports = _get_imports(main_file)
    assert "src.server" not in imports, "src/main.py must not import from src.server"


def test_main_imports_from_tts() -> None:
    """CLI module should import shared functions from tts module."""
    main_file = _SRC_DIR / "main.py"
    imports = _get_imports(main_file)
    assert "src.tts" in imports, "src/main.py should import from src.tts"


def test_server_imports_from_tts() -> None:
    """Server module should import shared functions from tts module."""
    server_file = _SRC_DIR / "server.py"
    if not server_file.exists():
        pytest.skip("src/server.py not yet created")
    imports = _get_imports(server_file)
    assert "src.tts" in imports, "src/server.py should import from src.tts"
