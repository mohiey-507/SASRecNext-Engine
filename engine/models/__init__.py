from .sasrec import SASRec
from .sasrecnext import SASRecNext

__all__ = [
    "SASRecNext",
    "SASRec",
    "MODEL_REGISTRY",
]

MODEL_REGISTRY = {
    "SASRec": SASRec,
    "SASRecNext": SASRecNext,
}
