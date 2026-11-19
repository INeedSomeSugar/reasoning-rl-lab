"""Record an observed resume; preserve the historical pause and frozen protocol."""
import argparse
import datetime as dt
import re
import sys

from reasoning_rl.runtime import ROOT, json_write, sha256_file
sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read


def main(family):
    exp = ROOT / "experiments" / family
    state = read(exp / "runner_state.json")
    assert state["status"] == "running" and state["stage"] == "main_training"
    units = read(exp / "training_units.json")
    done = sum(u["status"] == "first_success_fixed" for u in units.values())
    assert done >= 21 and not (exp / "fresh_development_opened.json").exists()
    resumes = sorted(exp.glob("controller_resume_*/resume_record.json"))
    assert resumes
    block = f"""<!-- LARGE_FILTER_CURRENT_BEGIN -->
## 当前执行：从已保存终点恢复固定预筛复验

`{family}`已按用户明确指令恢复，保留并核验此前21个成功终点，从R-j20261111-s71继续原冻结顺序；暂停历史收据保留。当前已完成{done}/35训练终点，正在执行`{state['phase']}`，新开发评估0/37。256题共同池、1024题fresh_development、三次K4筛选M=70/62/70和B/E/O/C=3600/7500/300/11400秒不变。全部35训练权重固定后才开启37对象评估。

U-s47-a0系统重启中断的原始输出和成本上下界保留；同S2/配置的唯一a1完整重试已成功。剩余任务沿用原预算与硬截止，已完成训练和探测不重跑；无精确sampler/dataloader中途续训声明。[实时运行状态](experiments/{family}/runner_state.json)、[冻结协议](experiments/{family}/protocol_frozen.json)、[历史暂停记录](reports/comparisons/{family}/paused_status.md)、[准备证据](reports/comparisons/{family}/preparation.md)、[复现入口](docs/REPRODUCING_FILTER_REPLICATION_20261110.md)。GSM8K test/SVAMP继续封存，完整对照结论尚未产生，远端未发布。
<!-- LARGE_FILTER_CURRENT_END -->"""
    hashes = {}
    for name in ("README.md", "PROJECT_CONTEXT.md", "HANDOFF.md"):
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        updated, count = re.subn(r"<!-- LARGE_FILTER_CURRENT_BEGIN -->.*?<!-- LARGE_FILTER_CURRENT_END -->", block, text, flags=re.S)
        assert count == 1
        path.write_text(updated, encoding="utf-8", newline="\n")
        hashes[name] = sha256_file(path)
    json_write(exp / "resumed_documentation_updated.json", {"recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runner_pid": state["runner_pid"], "completed_training_at_update": done, "fresh_evaluations": 0,
        "resume_record": resumes[-1].relative_to(ROOT).as_posix(), "files": hashes,
        "historical_pause_receipts_preserved": True, "original_deadline_unchanged": True})
    print({"state": "resumed", "training": done, "phase": state["phase"], "runner_pid": state["runner_pid"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    main(parser.parse_args().family)
