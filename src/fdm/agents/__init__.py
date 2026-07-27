from .debate import DebateConfig, run_debate, single_shot
from .schema import DebateResult, Turn, Verdict, normalize_suitability

__all__ = [
    "DebateConfig",
    "DebateResult",
    "Turn",
    "Verdict",
    "normalize_suitability",
    "run_debate",
    "single_shot",
]
