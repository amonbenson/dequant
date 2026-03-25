from ..config import CONFIG
from ..data.preprocessing import preprocess_egmd, preprocess_lmd


def run_preprocess():
    if CONFIG.data.egmd.enabled:
        preprocess_egmd()
    if CONFIG.data.lmd.enabled:
        preprocess_lmd()
