from .egmd import preprocess_egmd
from ..config import CONFIG


def preprocess():
    if CONFIG.data.egmd.enabled:
        preprocess_egmd()
