"""Tool registration interface, following the usual agent-framework pattern.

One registration carries both the metadata the model sees (description and
argument sample) and the executable handler. Built-in research tools register
through the same call, so adding a capability means registering a handler here
rather than editing a dispatch chain.
"""
from .. import db
from . import tools as builtin

_ORDER = []
_TOOLS = {}


def register_tool(name, handler, description=None, arguments=None, required=None, source="builtin"):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("工具需要非空名称")
    if not callable(handler):
        raise ValueError("工具注册需要可调用处理器")
    known = next((item for item in builtin.TOOLS if item["name"] == name), None) or {}
    description = description if description is not None else known.get("description", "")
    arguments = arguments if arguments is not None else known.get("arguments", {})
    required = list(required) if required is not None else list(arguments)
    entry = {"name": name, "description": description, "arguments": arguments,
             "required": required, "handler": handler, "source": source}
    _TOOLS[name] = entry
    if name not in _ORDER:
        _ORDER.append(name)
    db.tool_register(name, description, arguments, getattr(handler, "__qualname__", repr(handler)), source=source)
    return entry


def get(name):
    return _TOOLS.get(name)


def names():
    return list(_ORDER)


def tools():
    return [{key: value for key, value in _TOOLS[name].items() if key != "handler"} for name in _ORDER]


def dispatch(engine, name, args):
    entry = _TOOLS.get(name)
    if entry is None:
        raise ValueError(f"未开放工具 {name}；请从可用工具选择")
    return entry["handler"](engine, args)
