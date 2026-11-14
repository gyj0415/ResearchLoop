"""Restricted, disposable NumPy experiment worker. No model/network credentials."""
import ast
import builtins
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path

ALLOWED_IMPORTS = {"numpy", "math", "statistics"}
BANNED = {"open", "exec", "eval", "compile", "globals", "locals", "getattr", "setattr", "delattr", "vars", "dir", "input", "breakpoint", "help", "exit", "quit", "memoryview", "__import__"}
BANNED_ATTRIBUTES = {"load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt", "fromfile", "tofile", "memmap", "ctypes", "ctypeslib", "f2py", "lib", "random", "testing",
                     "fromregex", "dump", "dumps", "DataSource", "sys", "os", "modules", "builtins", "importlib", "subprocess", "io", "pathlib", "shutil", "pickle", "socket"}
BLOCKED_EVENTS = ("os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.startfile", "os.fork", "os.kill",
                  "subprocess.", "_winapi.", "socket.", "ctypes.", "winreg.", "os.remove", "os.unlink", "os.rename",
                  "os.replace", "os.rmdir", "os.mkdir", "os.chmod", "os.symlink", "os.link", "os.truncate",
                  "shutil.", "os.chdir", "urllib.", "http.", "webbrowser.")


def validate(code):
    if not isinstance(code, str) or len(code) > 14000:
        raise ValueError("实验代码为空或超过14000字符")
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name not in ALLOWED_IMPORTS for alias in node.names):
            raise ValueError(f"第{node.lineno}行：仅支持 numpy、math、statistics；数据请用load_csv/load_text")
        if isinstance(node, ast.ImportFrom) and (node.level or node.module not in ALLOWED_IMPORTS or any(a.name.startswith('_') or a.name == '*' or a.name in BANNED_ATTRIBUTES for a in node.names)):
            raise ValueError(f"第{node.lineno}行：不支持 from {node.module} import；仅可使用numpy/math/statistics，随机数直接用预置rng.normal或rng.uniform，不要导入随机模块")
        if isinstance(node, ast.Name) and (node.id in BANNED or node.id.startswith("__")):
            raise ValueError(f"不支持名称 {node.id}")
        rng_random = isinstance(node, ast.Attribute) and node.attr == "random" and isinstance(node.value, ast.Name) and node.value.id == "rng"
        if isinstance(node, ast.Attribute):
            owner = node.value.id if isinstance(node.value, ast.Name) else None
            # np.random.* is rejected whatever method follows it; name the fix the model actually needs.
            if node.attr == "random" and owner == "np":
                raise ValueError(f"第{node.lineno}行：不允许使用 np.random。命名空间已提供 rng（系统用隐藏种子初始化），随机数直接调用 rng.normal/rng.uniform/rng.integers；删除 np.random.*，不要自行创建生成器或设置种子，例如删除 `rng = np.random.default_rng()` 整行后直接用 rng。")
            if node.attr.startswith("_") or (node.attr in BANNED_ATTRIBUTES and not rng_random):
                raise ValueError(f"第{node.lineno}行：不允许属性 {node.attr}。实验接口只提供 np、rng、math、statistics 与 load_csv/load_text/emit_result。")
    return tree


def execute(payload):
    import numpy as np
    tree = validate(payload["code"])
    assets = payload.get("assets", {})
    captured = []
    def asset(identifier):
        if identifier not in assets:
            raise ValueError("未知材料ID")
        return Path(assets[identifier])
    def load_csv(identifier):
        with asset(identifier).open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    def load_text(identifier):
        return asset(identifier).read_text(encoding="utf-8-sig")[:200000]
    def emit_result(result):
        if not isinstance(result, dict) or not isinstance(result.get("metrics"), dict):
            raise ValueError("emit_result需要包含metrics字典")
        captured.append(result)
    def safe_import(name, *args, **kwargs):
        # numpy lazily imports its own numpy._core submodules on first use.
        if name not in ALLOWED_IMPORTS and not str(name).startswith("numpy."):
            raise ImportError(f"此模块未开放：{name}；实验只允许 numpy、math、statistics，随机数用提供的 rng")
        return builtins.__import__(name, *args, **kwargs)
    safe_names = "abs all any bool dict enumerate filter float int isinstance len list map max min next pow print range reversed round set slice sorted str sum tuple zip Exception ValueError TypeError ZeroDivisionError".split()
    safe = {name: getattr(builtins, name) for name in safe_names}
    safe["__import__"] = safe_import
    namespace = {"__builtins__": safe, "np": np, "math": math, "statistics": statistics,
                 "rng": np.random.default_rng(int(payload["seed"])), "PHASE": payload["phase"],
                 "load_csv": load_csv, "load_text": load_text, "emit_result": emit_result}
    exec(compile(tree, "<research-experiment>", "exec"), namespace, namespace)
    result = captured[-1] if captured else namespace.get("result")
    if not isinstance(result, dict) or not result.get("metrics"):
        raise ValueError('未返回结果；请调用emit_result({"metrics": {"指标": 数值}, "notes": "说明"})')
    if any(not isinstance(v, (int, float, np.number)) or isinstance(v, (bool, complex, np.complexfloating))
           or not math.isfinite(float(v)) for v in result["metrics"].values()):
        raise ValueError("metrics必须为有限实数，不接受NaN、复数、布尔值或文字")
    result["metrics"] = {str(k): float(v) for k, v in result["metrics"].items()}
    return result


def restrict(payload):
    """Process-wide audit hook: once installed it cannot be removed by experiment code."""
    import numpy  # noqa: F401  Import before the hook so its own module files load normally.
    import numpy.fft, numpy.linalg, numpy.polynomial, numpy.random  # noqa: E401,F401
    sys.dont_write_bytecode = True
    readable = [Path(p).resolve() for p in {sys.prefix, sys.base_prefix, sys.exec_prefix, Path(__file__).parent}]
    assets = {Path(p).resolve() for p in (payload.get("assets") or {}).values()}

    def readable_path(raw):
        try:
            path = Path(raw).resolve()
        except (TypeError, ValueError, OSError):
            return False
        return path in assets or any(path.is_relative_to(root) for root in readable)

    def hook(event, args):
        if event.startswith(BLOCKED_EVENTS):
            raise PermissionError(f"实验进程不允许 {event}")
        if event == "open":
            path, mode, flags = (list(args) + [None, None, None])[:3]
            if isinstance(path, int):
                return
            writes = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                     (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
            if writes or not readable_path(path):
                raise PermissionError("实验进程只能读取声明的材料，不能写文件或读取其他路径")

    sys.addaudithook(hook)


def run(payload):
    restrict(payload)
    return execute(payload)


if __name__ == "__main__":
    try:
        result = run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
        print("\nRESEARCH_RESULT=" + json.dumps(result, ensure_ascii=False, allow_nan=False))
    except Exception as error:
        print("\nRESEARCH_ERROR=" + type(error).__name__ + ": " + str(error))
        sys.exit(1)
