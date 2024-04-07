from pathlib import Path

root = Path(__file__).parent.parent
print(root)
DATA_PATH = root / "data/"
TRAINING_PATH = root / "outputs/training/"
EVAL_PATH = root / "outputs/results/"
