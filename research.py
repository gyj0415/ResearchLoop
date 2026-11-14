"""Command-line access to the same question-driven engine as the UI."""
import argparse
import json
from researchloop import storage
from researchloop.engine import core, history


def main():
    parser = argparse.ArgumentParser(description="ResearchLoop research agent")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    run = commands.add_parser("run")
    run.add_argument("--question", required=True)
    run.add_argument("--model")
    run.add_argument("--tokens", type=int, default=40000)
    run.add_argument("--seconds", type=int, default=600)
    run.add_argument("--steps", type=int, default=12)
    resume = commands.add_parser("resume")
    resume.add_argument("run_id")
    fork = commands.add_parser("fork")
    fork.add_argument("run_id")
    fork.add_argument("node_id")
    fork.add_argument("--question", default="")
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps([{"id": r["id"], "question": r.get("question"), "status": r["status"]}
                          for r in storage.list_runs()], ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        engine = core.create({"question": args.question, "model": args.model,
            "token_limit": args.tokens, "seconds_limit": args.seconds, "max_steps": args.steps})
    else:
        path = storage.run_path(args.run_id)
        run, folder = storage.read_json(path), path.parent
        if args.command == "fork":
            run, folder = history.fork(run, folder, args.node_id, args.question)
        engine = core.resume_paused(run, folder)
    result = engine.execute()
    print(json.dumps({key: result.get(key) for key in ("id", "status", "budget", "report", "stop_reason")}, ensure_ascii=False, indent=2))
    return 0 if result["status"] in ("completed", "paused") else 1


if __name__ == "__main__":
    raise SystemExit(main())
