"""Production post-filter parity via AST extraction.

The live ws_server module loads the GPU model on import, so it is never imported. Instead,
selected top-level pure functions/constants are parsed with ``ast`` and compiled in an
isolated namespace that only exposes a few stdlib modules and safe builtins. Anything that
is not provably self-contained is rejected with an explanation.
"""
import ast
import builtins
import collections
import functools
import hashlib
import itertools
import math
import re
import string
import unicodedata
from pathlib import Path

ALLOWED_MODULES = {"re": re, "unicodedata": unicodedata, "collections": collections,
                   "itertools": itertools, "math": math, "string": string, "functools": functools}
SAFE_BUILTINS = {n: getattr(builtins, n) for n in (
    "abs", "all", "any", "bool", "chr", "dict", "enumerate", "filter", "float", "frozenset", "int",
    "isinstance", "len", "list", "map", "max", "min", "ord", "range", "reversed", "round", "set",
    "sorted", "str", "sum", "tuple", "zip", "Exception", "ValueError", "TypeError", "KeyError",
    "IndexError")}


class PostfilterError(ValueError):
    pass


def _top_level(tree):
    top = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            top[node.name] = node
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            for t in node.targets:
                top[t.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            top[node.target.id] = node
    return top


def inspect_source(path):
    """List extractable top-level names without executing anything."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    top = _top_level(tree)
    return {"source": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "functions": sorted(n for n, v in top.items() if isinstance(v, ast.FunctionDef)),
            "assignments": sorted(n for n, v in top.items() if not isinstance(v, ast.FunctionDef))}


def _check_pure(node, allowed):
    local = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef,
                          ast.AsyncFunctionDef, ast.With, ast.AsyncWith, ast.Await)):
            raise PostfilterError(f"line {n.lineno}: {type(n).__name__} not allowed in extracted code")
        if isinstance(n, ast.FunctionDef):
            if n.decorator_list:
                raise PostfilterError(f"line {n.lineno}: decorated function {n.name!r} not extractable")
            local.add(n.name)
        if isinstance(n, ast.Attribute) and n.attr.startswith("_"):
            raise PostfilterError(f"line {n.lineno}: private/dunder attribute {n.attr!r} not allowed")
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            local.add(n.id)
        elif isinstance(n, ast.arg):
            local.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            local.add(n.name)
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in local | allowed:
            raise PostfilterError(f"line {n.lineno}: free name {n.id!r} is neither extracted, an allowed "
                                  f"stdlib module ({sorted(ALLOWED_MODULES)}) nor a safe builtin")


def extract(path, names):
    path = Path(path)
    src = path.read_bytes()
    tree = ast.parse(src.decode("utf-8"), filename=str(path))
    top = _top_level(tree)
    missing = [n for n in names if n not in top]
    if missing:
        raise PostfilterError(f"{path}: top-level names not found: {missing}")
    wanted = {id(top[n]) for n in names}
    selected = [node for node in tree.body if id(node) in wanted]
    allowed = set(SAFE_BUILTINS) | set(ALLOWED_MODULES) | set(names)
    for node in selected:
        _check_pure(node, allowed)
    ns = {"__builtins__": dict(SAFE_BUILTINS), **ALLOWED_MODULES}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), ns)
    return ns, {"source": str(path), "source_sha256": hashlib.sha256(src).hexdigest(), "names": list(names)}


class Postfilter:
    """Sequential filters. returns='text': f(text, *extra) -> text ('' / None = dropped);
    returns='drop_if_true': predicate f(text, *extra) -> True drops the utterance."""

    def __init__(self, steps=(), meta=None):
        self.steps = list(steps)
        self.meta = meta or {"status": "none"}

    def apply(self, text):
        for fn, returns, extra in self.steps:
            out = fn(text, *extra)
            if returns == "drop_if_true":
                text = "" if out else text
            elif out is None:
                text = ""
            elif isinstance(out, str):
                text = out
            else:
                raise PostfilterError(f"filter returned {type(out).__name__}, expected str/None")
            if not text:
                break
        return text


NOT_CONFIGURED = ("production post-filters (default blocklist, repetition detector) are NOT applied: "
                  "parity is UNVERIFIED. Run `postfilter-inspect`, then list the pure names/filters in the "
                  "profile. Stateless per-utterance calls are assumed.")


def load_postfilter(cfg, root):
    mode = (cfg or {}).get("mode", "none")
    if mode == "none":
        return Postfilter(meta={"status": "none", "note": (cfg or {}).get("note")})
    if mode != "ast_extract":
        raise PostfilterError(f"unknown postfilter mode {mode!r}")
    src = (Path(root) / cfg["source"]).resolve()
    filters = cfg.get("filters") or []
    if not filters:
        return Postfilter(meta={"status": "not_configured", "parity": "UNVERIFIED_GAP", "source": str(src),
                                "source_exists": src.is_file(), "limitation": NOT_CONFIGURED,
                                "profile_note": cfg.get("note")})
    ns, meta = extract(src, cfg["extract_names"])
    steps = []
    for f in filters:
        if f.get("returns", "text") not in ("text", "drop_if_true") or f["entry"] not in ns:
            raise PostfilterError(f"bad filter spec {f}")
        steps.append((ns[f["entry"]], f.get("returns", "text"), tuple(ns[a] for a in f.get("extra_args", []))))
    meta.update(status="extracted", parity="extracted_unverified", filters=filters,
                limitation="stateless per-utterance calls assumed; verify against production behaviour")
    return Postfilter(steps, meta)
