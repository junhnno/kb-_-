from .finance import attach_finance
from .loader import filter_segment, load_personas, sample_cohort, synthesize
from .schema import AGE_BANDS, FinanceProfile, Persona, Segment, age_band

__all__ = [
    "AGE_BANDS",
    "FinanceProfile",
    "Persona",
    "Segment",
    "age_band",
    "attach_finance",
    "filter_segment",
    "load_personas",
    "sample_cohort",
    "synthesize",
]
