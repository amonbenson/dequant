from ..data.preprocessing import preprocess_egmd
from ..config import CONFIG


def run_preprocess():
    if CONFIG.data.egmd.enabled:
        preprocess_egmd()
