"""Command-line access to the same research engine as the local UI."""
import argparse
import json
import time

from researchloop.dataset import prepare
from researchloop.runner import ResearchService
from researchloop.storage import list_runs, memory_state, promote, read_json, rollback, run_path


def main():
    parser = argparse.ArgumentParser(description="ResearchLoop CLI")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("status")
    run = commands.add_parser("run")
    run.add_argument("--task", choices=["A", "B", "C", "D", "E", "F"], default="C")
    run.add_argument("--mode", choices=["agent", "search", "carry"], default="agent")
    run.add_argument("--use-memory", action="store_true")
    run.add_argument("--experiments", type=int, default=4)
    run.add_argument("--tokens", type=int, default=16000)
    run.add_argument("--seconds", type=int, default=900)
    run.add_argument("--seed", type=int, default=42)
    publish = commands.add_parser("publish")
    publish.add_argument("run_id")
    activate = commands.add_parser("activate")
    activate.add_argument("version", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(print)
    elif args.command == "status":
        print(json.dumps({"runs": [{"id": r["id"], "status": r["status"], "mode": r["mode"]} for r in list_runs()], "memory": memory_state()}, ensure_ascii=False, indent=2))
    elif args.command == "publish":
        print(json.dumps(promote(args.run_id), ensure_ascii=False, indent=2))
    elif args.command == "activate":
        print(json.dumps(rollback(args.version), ensure_ascii=False, indent=2))
    elif args.command == "run":
        service = ResearchService()
        created = service.start({"task": args.task, "mode": args.mode, "use_memory": args.use_memory,
                                 "max_experiments": args.experiments, "token_limit": args.tokens,
                                 "seconds_limit": args.seconds, "seed": args.seed})
        emitted = 0
        while True:
            saved = read_json(run_path(created["id"]))
            for event in saved["events"][emitted:]:
                print(event["type"] + ": " + event["text"], flush=True)
            emitted = len(saved["events"])
            if saved["status"] not in ("running", "queued"):
                print(json.dumps({k: saved.get(k) for k in ("id", "status", "best_config", "budget", "stop_reason", "error")}, ensure_ascii=False, indent=2))
                return 0 if saved["status"] == "completed" else 1
            time.sleep(.4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
