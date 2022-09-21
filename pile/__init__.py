from .templates import Dataset
from .datasets import *

import logging
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(Path(__file__).parent / "pile.log"),
        logging.StreamHandler(),
    ],
)
