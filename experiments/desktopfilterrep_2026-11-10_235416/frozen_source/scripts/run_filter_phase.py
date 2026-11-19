"""Run one preregistered child with persisted process-level cost accounting."""
import argparse
from filter_common import run_phase

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    run_phase(args.label, command, timeout=args.timeout)
