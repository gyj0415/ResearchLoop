"""Workflow planning for a research session.

The planner proposes a stage list (role, goal, step budget); the orchestrator
executes the stages in order, each as a sub-agent with its own tools and model.
Roles are free-form within the known set; the plan is retried once with the
parse error quoted before falling back to the deterministic default, and the
fallback is recorded rather than hidden.
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
                           "why": {"type": "string"},
                           "max_steps": {"type": "integer"}},
            "required": ["role", "goal", "why"]}},
        "reason": {"type": "string"},
    },
    "required": ["stages"],
}

MAX_STAGES = 6
ROLE_ALIASES = {"规划": "planner", "计划": "planner", "实验": "experimenter", "复核": "verifier",
                "验证": "verifier", "报告": "reporter", "撰写": "reporter",
                "planner": "planner", "experimenter": "experimenter", "verifier": "verifier", "reporter": "reporter"}


def default_stages():
    return [
        {"role": "planner", "goal": "检查材料与文献，确认可行性并注册可证伪假设", "max_steps": 4,
         "why": "先确认证据与数据是否足以回答，避免在缺数据时用模拟或计数替代真实研究"},
        {"role": "experimenter", "goal": "为已注册假设运行真实实验并登记测量", "max_steps": 4,
         "why": "假设必须用可执行实验与真实测量检验，而不是由文本推断"},
        {"role": "verifier", "goal": "用新的隐藏种子独立复核冻结代码", "max_steps": 2,
         "why": "独立重跑能暴露偶然结果，复核不通过是有效负结果"},
        {"role": "reporter", "goal": "依据复核结果形成结论、局限与方法经验", "max_steps": 2,
         "why": "只报告复核过的证据，并写清证据缺口与适用边界"},
    ]


def validate(stages):
    if not isinstance(stages, list) or not 1 <= len(stages) <= MAX_STAGES:
        raise ValueError(f"工作流需要 1–{MAX_STAGES} 个阶段")
    cleaned = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError(f"第{index + 1}阶段必须是对象（role/goal/max_steps），收到 {type(stage).__name__}")
        role = ROLE_ALIASES.get(str(stage.get("role", "")).strip(), "")
        goal = str(stage.get("goal", "")).strip()
        if role not in roles.ROLES:
            raise ValueError(f"第{index + 1}阶段角色无效：{stage.get('role')}")
        if not goal or len(goal) > 400:
            raise ValueError(f"第{index + 1}阶段缺少目标描述")
        steps = stage.get("max_steps")
        steps = int(steps) if isinstance(steps, int) else 3
        cleaned.append({"role": role, "goal": goal, "max_steps": max(1, min(6, steps)),
                        "why": str(stage.get("why") or "").strip()[:400]})
    return cleaned



def design(engine, args):
    """The design_workflow tool: the agent proposes stages and must justify them."""
    raw = args.get("stages")
    stages = validate(raw)
    reason = str(args.get("reason") or "").strip()
    if not reason:
        raise ValueError("design_workflow 需要整体设计理由 reason")
    for stage, source in zip(stages, raw):
        if not stage["why"]:
            raise ValueError(f"阶段 {stage['role']} 缺少设计理由 why")
    workflow = {"source": "agent_tool", "reason": reason[:600], "stages": stages,
                "planned_at": engine.run.get("updated_at")}
    engine.run["workflow"] = workflow
    engine.event("workflow", "工作流设计完成：" + " → ".join(stage["role"] for stage in stages),
                 status="completed", workflow=workflow)
    return workflow


def plan(engine):
    """Ask the planner role for a workflow, retry once, then fall back deterministically."""
    run = engine.run
    prompt = {"question": engine.question_text(), "assets": run["assets"], "budget": engine.budget.snapshot(),
              "roles": roles.describe(),
              "instruction": "只输出stages数组：为完成这个研究问题选择1–6个阶段，每个阶段给出role、goal、max_steps。"
                             "定性或文献问题可以没有实验阶段；需要数值实验时必须有experimenter，复核与报告应在实验之后。"
                             "每个阶段必须写 why：为什么需要这个阶段、它解决什么问题。阶段目标不能预设尚未核实的结论；先验证问题前提或出处。不要编造材料中没有的数据。"}
    attempts = []
    stages = None
    source = "default"
    reason = ""
    for purpose in ("workflow_plan", "workflow_plan_retry"):
        try:
            extra = "" if not attempts else f"\n上次输出无效：{attempts[-1]}。必须严格按 schema 返回对象数组。"
            parsed, _ = engine.complete(roles.ROLES["planner"]["system"] + extra, prompt, SCHEMA,
                                        purpose=purpose, role="planner")
            stages = validate(parsed.get("stages"))
            source = "model" if purpose == "workflow_plan" else "model_retry"
            reason = str(parsed.get("reason", ""))[:200]
            break
        except Exception as error:
            attempts.append(str(error)[:200])
    if stages is not None:
        defaults = {
            "planner": "先确认证据与可行性，避免在缺数据时用计数或模拟替代真实研究",
            "experimenter": "用可执行实验检验假设，而不是由文本推断结论",
            "verifier": "独立重跑暴露偶然结果，复核不通过也是有效结果",
            "reporter": "只报告复核过的证据，并写清证据缺口与适用边界",
        }
        for stage in stages:
            if not stage.get("why"):
                stage["why"] = defaults.get(stage["role"], "该阶段是该方法链条上的必要环节")
        if not reason:
            reason = "规划器未给出整体理由；各阶段理由按角色职责补齐"
    if stages is None:
        stages = default_stages()
        source = "default"
        reason = "模型工作流两次不可用，使用默认四阶段：" + "；".join(attempts)
    workflow = {"source": source, "reason": reason, "stages": stages,
                "attempts": attempts, "planned_at": run.get("updated_at")}
    run["workflow"] = workflow
    engine.event("workflow", f"工作流已生成（{source}）：" + " → ".join(stage["role"] for stage in stages),
                 status="completed", level="info" if source.startswith("model") else "warning", workflow=workflow)
    return workflow

from . import registry

registry.register_tool("design_workflow", design,
                       description="设计研究的工作流：给出每个阶段角色、目标、步数上限与设计理由，供编排器执行。",
                       arguments={"reason": "整体设计理由",
                                  "stages": [{"role": "planner", "goal": "阶段目标", "why": "为什么需要这个阶段",
                                              "max_steps": 3}]},
                       required=["reason", "stages"])
