import json
import pytest
from reasoning_rl import training
from reasoning_rl.runtime import sha256_file


def test_untrained_control_cannot_be_mislabeled_as_real_sft_or_public_initialization(tmp_path):
    path=tmp_path/"rrl_checkpoint.json"
    path.write_text(json.dumps({"stage":"untrained_lora_control"}))
    with pytest.raises(ValueError,match="stage sft"):
        training._starting_checkpoint_metadata(tmp_path)
    with pytest.raises(ValueError,match="stage public_posttrained_initialization"):
        training._starting_checkpoint_metadata(tmp_path,public_start=True)


def test_public_initialization_requires_explicit_zero_updates_and_unchanged_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(training,"ROOT",tmp_path)
    (tmp_path/"data").mkdir()
    (tmp_path/"data/manifest.json").write_text("data provenance")
    (tmp_path/"receipt.json").write_text("verified public bytes")
    meta={"stage":"public_posttrained_initialization","explicitly_not_sft":True,"optimizer_steps":0,
          "source_verification_receipt":"receipt.json","source_verification_receipt_sha256":sha256_file(tmp_path/"receipt.json"),
          "data_manifest_sha256":sha256_file(tmp_path/"data/manifest.json")}
    path=tmp_path/"rrl_checkpoint.json"
    path.write_text(json.dumps(meta))
    assert training._starting_checkpoint_metadata(tmp_path,True)[1]=="public_posttrained_initialization"
    meta["optimizer_steps"]=1
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match="untrained"):
        training._starting_checkpoint_metadata(tmp_path,True)
    meta["optimizer_steps"]=0
    path.write_text(json.dumps(meta))
    (tmp_path/"receipt.json").write_text("changed")
    with pytest.raises(ValueError,match="receipt changed"):
        training._starting_checkpoint_metadata(tmp_path,True)
