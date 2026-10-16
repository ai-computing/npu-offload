#!/usr/bin/env python3
"""Record research progress for research-hub.

Updates the STATUS.md header and log, then appends one event to .research/events.jsonl.
The hub reads both through Orca and syncs them to Plane, so this script needs no network.
Python 3.6+, stdlib only.

    python3 .research/report.py --task "QKV optimization" --result "+8.3% tok/s" --next "tile=256"
"""
import argparse
import datetime
import json
import os
import re
import socket
import subprocess
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATUS = os.path.join(ROOT, "STATUS.md")
EVENTS = os.path.join(ROOT, ".research", "events.jsonl")

HEADER_BEGIN = "<!-- research-hub:header:begin"
HEADER_END = "<!-- research-hub:header:end -->"
LOG_MARK = "<!-- research-hub:log"
STATES = ("active", "blocked", "idle", "done", "unknown")


def git(*args):
    try:
        out = subprocess.check_output(("git",) + args, cwd=ROOT, stderr=subprocess.DEVNULL)
        return out.decode("utf-8", "replace").strip()
    except Exception:
        return ""


def detect_agent():
    if os.environ.get("CLAUDECODE") or os.environ.get("CLAUDE_CODE_ENTRYPOINT"):
        return "claude"
    if any(k.startswith("CODEX_") for k in os.environ):
        return "codex"
    return os.environ.get("RESEARCH_AGENT", "human")


def one_line(text):
    return re.sub(r"\s+", " ", text or "").strip()


def update_header(text, fields):
    begin = text.find(HEADER_BEGIN)
    end = text.find(HEADER_END)
    if begin < 0 or end < 0:
        raise SystemExit("STATUS.md has no research-hub header block; run `researchctl init` from the hub")
    block = text[begin:end]
    for key, value in fields.items():
        line = "- **%s**: %s" % (key, value)
        pattern = re.compile(r"^- \*\*%s\*\*:.*$" % re.escape(key), re.M)
        if pattern.search(block):
            block = pattern.sub(lambda _m: line, block, count=1)
        else:
            block = block.rstrip("\n") + "\n" + line + "\n"
    return text[:begin] + block + text[end:]


def insert_log(text, entry):
    pos = text.find(LOG_MARK)
    if pos < 0:
        return text.rstrip("\n") + "\n\n## Log\n\n" + entry
    eol = text.find("\n", pos)
    eol = len(text) if eol < 0 else eol + 1
    return text[:eol] + "\n" + entry + text[eol:]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="Plane work item ID (FURIOSA-12) or a short task title")
    p.add_argument("--result", required=True, help="what was found or done, with numbers")
    p.add_argument("--next", default="", help="the next concrete action")
    p.add_argument("--state", choices=STATES, help="project state (default: active, or idle with --done)")
    p.add_argument("--done", action="store_true", help="the task is finished")
    p.add_argument("--blocker", default="", help="why work is blocked (implies --state blocked)")
    p.add_argument("--metric", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--experiment", action="append", default=[], metavar="PATH", help="experiment record(s)")
    p.add_argument("--label", action="append", default=[], metavar="NAME",
                   help="Plane label for the task (repeatable); see .research/TASKS.md for the list")
    p.add_argument("--note", default="", help="extra detail for the log and the Plane comment")
    p.add_argument("--agent", default=None, help="who is reporting (default: detected)")
    args = p.parse_args()

    if not os.path.exists(STATUS):
        raise SystemExit("STATUS.md not found at %s" % STATUS)

    metrics = {}
    for item in args.metric:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit("--metric expects KEY=VALUE, got %r" % item)
        metrics[key.strip()] = value.strip()

    state = args.state or ("blocked" if args.blocker else ("idle" if args.done else "active"))
    now = datetime.datetime.now().astimezone()
    stamp = now.isoformat(timespec="minutes").replace("T", " ")
    agent = args.agent or detect_agent()
    host = socket.gethostname()
    commit = git("rev-parse", "--short", "HEAD")
    if commit and git("status", "--porcelain"):
        commit += "-dirty"
    branch = git("rev-parse", "--abbrev-ref", "HEAD")

    with open(STATUS, encoding="utf-8") as f:
        text = f.read()

    text = update_header(text, {
        "State": state,
        "Updated": stamp,
        "Agent": "%s@%s" % (agent, host),
        "Task": one_line(args.task) + (" (done)" if args.done else ""),
        "Last result": one_line(args.result),
        "Next": one_line(args.next) or "-",
        "Blockers": one_line(args.blocker) or "none",
    })

    lines = ["### %s — %s%s" % (stamp, one_line(args.task), " ✅" if args.done else ""), ""]
    lines.append("- Result: %s" % one_line(args.result))
    if args.next:
        lines.append("- Next: %s" % one_line(args.next))
    if args.blocker:
        lines.append("- Blocker: %s" % one_line(args.blocker))
    if metrics:
        lines.append("- Metrics: %s" % ", ".join("%s=%s" % kv for kv in metrics.items()))
    if args.experiment:
        lines.append("- Experiments: %s" % ", ".join(args.experiment))
    if args.note:
        lines.append("- Note: %s" % one_line(args.note))
    lines.append("- By: %s@%s%s" % (agent, host, (" · %s@%s" % (branch, commit)) if commit else ""))
    text = insert_log(text, "\n".join(lines) + "\n")

    with open(STATUS, "w", encoding="utf-8") as f:
        f.write(text)

    event = {
        "id": uuid.uuid4().hex,
        "ts": now.isoformat(timespec="seconds"),
        "task": args.task.strip(),
        "result": args.result.strip(),
        "next": args.next.strip(),
        "state": state,
        "done": args.done,
        "blocker": args.blocker.strip(),
        "metrics": metrics,
        "experiments": args.experiment,
        "labels": [l.strip() for l in args.label if l.strip()],
        "note": args.note.strip(),
        "agent": agent,
        "host": host,
        "branch": branch,
        "commit": commit,
    }
    os.makedirs(os.path.dirname(EVENTS), exist_ok=True)
    with open(EVENTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    print("reported: [%s] %s -> %s" % (state, args.task, args.result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
