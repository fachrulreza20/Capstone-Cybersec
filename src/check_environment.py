import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.json"


def main():
    print("=== Capstone Environment Check ===")
    print(f"Python executable : {sys.executable}")
    print(f"Project root      : {PROJECT_ROOT}")
    print()

    print("=== Core Dependencies ===")
    print(f"NumPy        : {np.__version__}")
    print(f"pandas       : {pd.__version__}")
    print(f"scikit-learn : {sklearn.__version__}")
    print(f"matplotlib   : {matplotlib.__version__}")
    print(f"joblib       : {joblib.__version__}")
    print()

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    print("=== Project Configuration ===")
    print(f"Project          : {config['project_name']}")
    print(f"Domain           : {config['domain']}")
    print(f"Random seed      : {config['random_seed']}")
    print(f"Roles            : {', '.join(config['roles'])}")
    print(f"Observation unit : {config['observation_unit']}")
    print(f"Main model       : {config['main_model']}")
    print()
    print("Environment check completed successfully.")


if __name__ == "__main__":
    main()