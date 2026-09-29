"""Decode BFCL prompting-mode Python calls."""

from __future__ import annotations

import ast
import re

_FENCE = re.compile(r"```(?:python|py)?\s*([\s\S]*?)```", re.IGNORECASE)
_TOOL = re.compile(r"<TOOLCALL>([\s\S]*?)</TOOLCALL>", re.IGNORECASE)


def decode_calls(text: str, functions: list[dict] | None = None) -> list[dict] | None:
    """Return [{name: {arg: value}}] or None when the text is not a call list."""
    candidates = []
    candidates.extend(_FENCE.findall(text or ""))
    candidates.extend(_TOOL.findall(text or ""))
    candidates.append(text or "")
    found_empty = False
    for candidate in candidates:
        parsed = _parse_any(candidate)
        if parsed:
            return _bind(parsed, functions or [])
        if parsed == []:
            found_empty = True
    if found_empty:
        return []
    return None


def _parse_any(text: str) -> list[tuple[str, dict]] | None:
    text = (text or "").strip()
    if not text:
        return None
    variants = [text, _relax(text), _strip_commas(_relax(text))]
    found_empty = False
    for variant in variants:
        parsed = _parse_variant(variant)
        if parsed:
            return parsed
        if parsed == []:
            found_empty = True
    return [] if found_empty else None


def _parse_variant(text: str) -> list[tuple[str, dict]] | None:
    spans = _bracket_spans(text)
    chunks = [text, *spans[::-1]]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        chunks.append(lines[-1])
    found_empty = False
    for chunk in chunks:
        parsed = _parse_chunk(chunk.strip())
        if parsed:
            return parsed
        if parsed == []:
            found_empty = True
    return [] if found_empty else None


def _parse_chunk(chunk: str) -> list[tuple[str, dict]] | None:
    if not chunk:
        return None
    for mode in ("eval", "exec"):
        try:
            tree = ast.parse(chunk, mode=mode)
        except SyntaxError:
            continue
        node = tree.body if mode == "eval" else None
        if mode == "exec":
            exprs = [item.value for item in tree.body if isinstance(item, ast.Expr)]
            if len(exprs) != 1:
                continue
            node = exprs[0]
        parsed = _from_expr(node)
        if parsed is not None:
            return parsed
    return None


def _from_expr(node) -> list[tuple[str, dict]] | None:
    if isinstance(node, ast.List):
        if not node.elts:
            return []
        calls = []
        for elt in node.elts:
            if not isinstance(elt, ast.Call):
                return None
            decoded = _decode_call(elt)
            if decoded is None:
                return None
            calls.append(decoded)
        return calls
    if isinstance(node, ast.Call):
        decoded = _decode_call(node)
        return [decoded] if decoded else None
    return None


def _decode_call(node: ast.Call) -> tuple[str, dict] | None:
    name = _func_name(node.func)
    if not name:
        return None
    try:
        params: dict = {}
        positional = []
        for arg in node.args:
            positional.append(_literal(arg))
        for keyword in node.keywords:
            if keyword.arg is None:
                return None
            params[keyword.arg] = _literal(keyword.value)
    except (ValueError, TypeError):
        return None
    if positional:
        params["__pos__"] = positional
    return name, params


def _func_name(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _func_name(node.value)
        if not parent:
            return None
        return parent + "." + node.attr
    return None


def _literal(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal(node.operand)
        if isinstance(value, (int, float)):
            return -value
        raise ValueError("unary")
    if isinstance(node, ast.List) or isinstance(node, ast.Tuple):
        return [_literal(elt) for elt in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(key): _literal(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.Name) and node.id in {"True", "False", "None"}:
        return {"True": True, "False": False, "None": None}[node.id]
    raise ValueError("not a literal")


def _bind(calls: list[tuple[str, dict]], functions: list[dict]) -> list[dict]:
    schemas = {item["name"]: item for item in functions}
    bound = []
    for name, params in calls:
        params = dict(params)
        positional = params.pop("__pos__", [])
        if positional:
            schema = schemas.get(name) or {}
            properties = list(((schema.get("parameters") or {}).get("properties") or {}))
            required = list(((schema.get("parameters") or {}).get("required") or []))
            order = required + [name_ for name_ in properties if name_ not in required]
            for index, value in enumerate(positional):
                if index >= len(order):
                    break
                params.setdefault(order[index], value)
        bound.append({name: params})
    return bound


def _relax(text: str) -> str:
    text = re.sub(r"\btrue\b", "True", text)
    text = re.sub(r"\bfalse\b", "False", text)
    text = re.sub(r"\bnull\b", "None", text)
    return text


def _strip_commas(text: str) -> str:
    return re.sub(r",(\s*[\]\)])", r"\1", text)


def _bracket_spans(text: str) -> list[str]:
    spans = []
    index = 0
    length = len(text)
    while index < length:
        if text[index] != "[":
            index += 1
            continue
        depth = 0
        quote = ""
        cursor = index
        while cursor < length:
            char = text[cursor]
            if quote:
                if char == "\\":
                    cursor += 2
                    continue
                if char == quote:
                    quote = ""
                cursor += 1
                continue
            if char in {"'", '"'} :
                quote = char
                cursor += 1
                continue
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    spans.append(text[index : cursor + 1])
                    break
            cursor += 1
        index = cursor + 1 if cursor >= index else index + 1
    return spans
