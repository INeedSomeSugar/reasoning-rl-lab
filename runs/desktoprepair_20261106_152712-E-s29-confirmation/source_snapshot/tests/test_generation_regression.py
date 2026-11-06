import pytest


def test_explicit_greedy_survives_posttrained_model_sampling_defaults():
    pytest.importorskip("torch")
    from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=16)
    model = Qwen3ForCausalLM(config)
    model.generation_config = GenerationConfig(do_sample=True, temperature=0.6, top_k=20, top_p=0.95)
    explicit = GenerationConfig(do_sample=False, max_new_tokens=4, eos_token_id=2, pad_token_id=0)
    prepared, _ = model._prepare_generation_config(explicit, use_model_defaults=False, do_sample=False)
    assert prepared.do_sample is False
    assert prepared.max_new_tokens == 4


def test_prompt_does_not_invite_copying_literal_answer_placeholder():
    from reasoning_rl.modeling import PROMPT
    assert "<numeric answer>" not in PROMPT
    assert "{question}" in PROMPT
