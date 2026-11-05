"""Conservative numeric final-answer extraction; no model or paid judge."""
import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from decimal import Decimal, InvalidOperation

SCORER_VERSION = "numeric_final_v2"

NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:\s*/\s*[+-]?\d+(?:\.\d+)?)?"
_numeric = re.compile(rf"^\s*\$?\s*({NUMBER})\s*\$?\s*[.]?\s*$")


def canonical_number(value: str) -> Fraction | None:
    value = str(value).strip().replace("−", "-")
    match = _numeric.fullmatch(value)
    if not match:
        return None
    try:
        parts = match.group(1).replace(",", "").replace(" ", "").split("/")
        result = Fraction(Decimal(parts[0]))
        if len(parts) == 2:
            result /= Fraction(Decimal(parts[1]))
        return result
    except (ValueError, ZeroDivisionError, InvalidOperation, OverflowError):
        return None


@dataclass(frozen=True)
class Score:
    correct: bool
    format_ok: bool
    parse_ok: bool
    extracted: str | None
    parse_method: str

    def as_dict(self):
        return asdict(self)


def extract_answer(text: str) -> tuple[str | None, str]:
    """Prefer the last explicit marker; never grab an arbitrary incidental number."""
    text = text.replace("−", "-").strip()
    # Ignore thought content if a caller inadvertently enables thinking.
    text = text.rsplit("</think>", 1)[-1].strip()
    lines = text.splitlines()
    def single_number(region):
        matches = re.findall(rf"(?<![\w.])({NUMBER})(?![\w.])", region)
        return matches[0] if len(matches) == 1 and canonical_number(matches[0]) is not None else None
    if "####" in text:
        # A final-answer section may place the number on a later line or wrap it in Markdown.
        # Exactly one explicit number is required; calculations/alternatives stay ambiguous.
        region = text.rsplit("####", 1)[1].strip()
        answer = single_number(region)
        return (answer, "hash_region") if answer is not None else (None, "invalid_hash")
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        answer = boxed[-1].strip()
        return (answer, "boxed") if canonical_number(answer) is not None else (None, "invalid_boxed")
    last = lines[-1].strip() if lines else ""
    plain_last = last.replace("**", "").replace("__", "")
    if re.match(r"(?i)^(?:final\s+)?answer\s*:", plain_last):
        answer = single_number(plain_last.split(":", 1)[1])
        return (answer, "answer_line") if answer is not None else (None, "invalid_answer_line")
    return (last, "standalone") if canonical_number(last) is not None else (None, "missing")


def score_answer(text: str, gold: str) -> Score:
    expected = canonical_number(gold)
    if expected is None:
        raise ValueError(f"Invalid gold numeric answer: {gold!r}")
    answer, method = extract_answer(text)
    value = canonical_number(answer) if answer is not None else None
    # Format is an independent metric, never an additional reward term.
    last = text.strip().splitlines()[-1] if text.strip() else ""
    format_ok = bool(re.fullmatch(rf"\s*####\s+{NUMBER}\s*", last))
    return Score(value == expected, format_ok, value is not None, answer, method)


def correctness_reward(completions, answer, **kwargs) -> list[float]:
    def content(c):
        return c[0]["content"] if isinstance(c, list) else c
    return [float(score_answer(content(c), a).correct) for c, a in zip(completions, answer, strict=True)]


def group_advantages(rewards: list[float], scale: str = "group") -> list[float]:
    """Readable reference for TRL's sample-standard-deviation normalization."""
    if len(rewards) < 2:
        raise ValueError("GRPO requires at least two completions per prompt")
    mean = sum(rewards) / len(rewards)
    std = (sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)) ** 0.5
    denom = std + 1e-4 if scale == "group" else 1.0
    if scale not in ("group", "none"):
        raise ValueError(scale)
    return [(r - mean) / denom for r in rewards]
