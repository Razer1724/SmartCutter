"""Train v9: python train_v9.py MANIFEST --config configs/v9.json --output runs/v9."""

import sys

from smartcutter.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
