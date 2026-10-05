"""Read-only legacy extraction and optional-dependency checks; no import caches."""
import ast
import importlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]

def available(*names):
    try:
        for name in names:
            importlib.import_module(name)
        return True
    except Exception:
        return False

def legacy_nodes(relative, names, namespace):
    path = REPO / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module = ast.parse("")
    module.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    found = {node.name for node in module.body}
    if found != set(names):
        raise AssertionError("Missing legacy definitions: " + str(set(names) - found))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace
