import logging

from ..config import CONFIG
from ..data.converters.hov_converter import HOVConverter, HOVConverterConfig

logger = logging.getLogger("cli")


def create_converter():
    return HOVConverter(
        HOVConverterConfig(
            steps_per_beat=CONFIG.model.drums.steps_per_beat,
            categories=CONFIG.model.drums.categories,
        )
    )
