"""Update navigation only after observing the immutable training-to-evaluation gate."""
import argparse
import datetime as dt
import re
import sys

from reasoning_rl.runtime import ROOT, json_write, sha256_file
sys.path.insert(0, str(ROOT / "scripts"))
from large_filter_common import read


def main(family):
    exp = ROOT / "experiments" / family
    lock = read(exp / "all_training_endpoints_fixed.json")
    opened = read(exp / "fresh_development_opened.json")
    assert lock["all35_successful"] and len(lock["objects"]) == 37
    assert opened["endpoint_lock_sha256"] == sha256_file(exp / "all_training_endpoints_fixed.json")
    assert len(list((exp / "completed_training_audits").glob("*.json"))) == 35
    evaluations = read(exp / "evaluation_units.json")
    n = sum(item["status"] == "first_success_fixed" for item in evaluations.values())
    phase = read(exp / "runner_state.json").get("phase")
    block = f"""<!-- LARGE_FILTER_CURRENT_BEGIN -->
## 当前评估：35个训练终点已固定

`{family}`的35/35训练终点全部完成并通过独立CPU证据审计，U/F/R实际终点数5/15/15；35个检查点哈希已统一登记，A/S2与训练对象共37份权重固定后才开启1024题fresh_development。当前独立新进程评估完成{n}/37，正在执行`{phase}`。全部评估结束前不写主准确率结论，后续不追加训练或更换终点。

256题共同池、K4固定集合M=70/62/70及B/E/O/C=3600/7500/300/11400秒均保持冻结。U-s47-a0系统重启中断证据及成本界保留，唯一a1完整重试成功；用户保存后暂停和继续的历史收据保留。已补齐含中断尝试的物理账本、未知精确值与失败成本界，主统计/阈值不变，未延长原硬截止。[实时状态](experiments/{family}/runner_state.json)、[全部权重锁](experiments/{family}/all_training_endpoints_fixed.json)、[独立锁核验](experiments/{family}/all35_lock_independent_verification.json)、[冻结协议](experiments/{family}/protocol_frozen.json)、[准备证据](reports/comparisons/{family}/preparation.md)、[复现入口](docs/REPRODUCING_FILTER_REPLICATION_20261110.md)。本集是原train后置隔离开发子集，GSM8K test/SVAMP继续封存，远端未发布。
<!-- LARGE_FILTER_CURRENT_END -->"""
    hashes = {}
    for name in ("README.md", "PROJECT_CONTEXT.md", "HANDOFF.md"):
        path = ROOT / name
        text, count = re.subn(r"<!-- LARGE_FILTER_CURRENT_BEGIN -->.*?<!-- LARGE_FILTER_CURRENT_END -->",
            block, path.read_text(encoding="utf-8"), flags=re.S)
        assert count == 1
        path.write_text(text, encoding="utf-8", newline="\n")
        hashes[name] = sha256_file(path)
    json_write(exp / "evaluation_documentation_updated.json", {"recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "training_fixed": 35, "fresh_evaluations_at_update": n, "phase": phase, "files": hashes,
        "no_prediction_contents_or_accuracies_read": True})
    print({"training": 35, "evaluation": n, "phase": phase, "post_training_stage": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    main(parser.parse_args().family)
