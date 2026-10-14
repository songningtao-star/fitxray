"""让 `python -m fitxray` 能跑起来。逻辑全在 cli.py，这里只是个转发。"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
