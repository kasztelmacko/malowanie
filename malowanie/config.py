from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import sys

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTLINES_DATA_DIR = DATA_DIR / "outlines"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

## PainterProcessConfig
SCALED_MAX_WIDTH = 1024

## PaintByNumberProcessConfig
N_COLORS = 20
DENOSING_KERNEL_SIZE = 9
# Font configuration
FONT_COLOR = (180, 180, 180)
FONT_SCALE = 0.2
FONT_THICKNESS = 1

MIN_REGION_SIZE = 100


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
