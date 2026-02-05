from ..config import CONFIG
from ..data.preprocessing import preprocess_egmd


def run_preprocess():
    if CONFIG.data.egmd.enabled:
        preprocess_egmd()
