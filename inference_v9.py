"""Run v9: python inference_v9.py INPUT --checkpoint runs/v9/best.pt --output infer_output."""

import sys

from smartcutter.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["infer", *sys.argv[1:]]))
