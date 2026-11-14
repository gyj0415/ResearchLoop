"""Workflow planning for a research session.

The planner proposes a stage list (role, goal, step budget); the orchestrator
executes the stages in order, each as a sub-agent with its own tools and model.
If the planner output is unusable the deterministic default workflow is used and
the fallback is recorded, never silently.
"""
import json

from . import roles

SCHEMA = {
    "type": "object",
    "properties": {
        "stages": {"type": "array", "items": {
            "type": "object",
            "properties": {"role": {"type": "string", "enum": list(roles.ROLES)},
                           "goal": {"type": "string"},
                           "max_steps": {"type": "integer"}},
            "required": ["role", "goal"]}},
        "reason": {"type": "string"},
    },
    "required": ["stages"],
}

MAX_STAGES = 6


def default_stages():
    return [
        {"role": "planner", "goal": "检查材料与文献，确认可行性并注册可证伪假设", "max_steps": 4},
        {"role": "experimenter", "goal": "为已注册假设运行真实实验并登记测量", "max_steps": 4},
        {"role": "verifier", "goal": "用新的隐藏种子独立复核冻结代码", "max_steps": 2},
        {"role": "reporter", "goal": "依据复核结果形成结论、局限与方法经验", "max_steps": 2},
    ]


def validate(stages):
    if not isinstance(stages, list) or not 2 <= len(stages) <= MAX_STAGES:
        raise ValueError(f"工作流需要 2–{MAX_STAGES} 个阶段")
    cleaned = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError(f"第{index + 1}阶段必须是对象（role/goal/max_steps），收到 {type(stage).__name__}")
        role = str(stage.get("role", "")).strip()
        goal = str(stage.get("goal", "")).strip()
        if role not in roles.ROLES:
            raise ValueError(f"第{index + 1}阶段角色无效：{role}")
        if not goal or len(goal) > 400:
            raise ValueError(f"第{index + 1}阶段缺少目标描述")
        steps = stage.get("max_steps")
        steps = int(steps) if isinstance(steps, int) else 3
        cleaned.append({"role": role, "goal": goal, "max_steps": max(1, min(6, steps))})
    seen = {stage["role"] for stage in cleaned}
    if "experimenter" not in seen:
        raise ValueError("工作流至少需要一个实验阶段")
    return cleaned


def plan(engine):
    """Ask the planner role for a workflow; fall back deterministically."""
    run = engine.run
    prompt = {"question": run["question"], "assets": run["assets"], "budget": engine.budget.snapshot(),
              "roles": roles.describe(),
              "instruction": "只输出stages数组：为完成这个研究问题选择2–6个阶段，每个阶段给出role、goal、max_steps。"
                             "必须包含至少一个experimenter阶段；复核和报告阶段应跟在实验之后。不要编造材料中没有的数据。"}
    try:
        parsed, item = engine.complete(roles.ROLES["planner"]["system"], prompt, SCHEMA, purpose="workflow_plan",
                                       role="planner")
        stages = validate(parsed.get("stages"))
        source = "model"
        reason = str(parsed.get("reason", ""))[:200]
    except Exception as error:
        stages = default_stages()
        source = "default"
        reason = f"模型工作流不可用，使用默认四阶段：{error}"
    workflow = {"source": source, "reason": reason, "stages": stages, "planned_at": engine.run.get("updated_at")}
    run["workflow"] = workflow
    engine.event("workflow", f"工作流已生成（{source}）：" + " → ".join(stage["role"] for stage in stages),
                 status="completed", level="info" if source == "model" else "warning", workflow=workflow)
    return workflow
