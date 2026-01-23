from .egmd import preprocess_egmd
from ..config import CONFIG


def preprocess():
    print(CONFIG)
    if CONFIG.preprocess.egmd.enabled:
        preprocess_egmd()
