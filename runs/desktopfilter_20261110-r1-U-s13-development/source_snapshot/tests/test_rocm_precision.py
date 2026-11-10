from argparse import Namespace
import pytest
from reasoning_rl.cli import read_config


@pytest.mark.parametrize("precision", ["float16", "bfloat16", "float32"])
def test_rocm_explicit_precision_is_not_silently_forced_to_fp16(tmp_path, precision):
    config = tmp_path / "rocm.yaml"
    config.write_text(f"dtype: {precision}\nquantization: none\n", encoding="utf-8")
    resolved = read_config(config, Namespace(hardware="rocm"))
    assert resolved["dtype"] == precision
    assert resolved["quantization"] == "none"


def test_rocm_does_not_inherit_laptop_nf4_compute_precision(tmp_path):
    config = tmp_path / "laptop.yaml"
    config.write_text("dtype: bfloat16\nquantization: nf4\n", encoding="utf-8")
    resolved = read_config(config, Namespace(hardware="rocm"))
    assert (resolved["dtype"], resolved["quantization"]) == ("float16", "none")
