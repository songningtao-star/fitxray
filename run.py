#!/usr/bin/env python
"""
一个薄薄的包装，让不装包也能直接跑。

    python run.py
    python run.py --html report.html
    python run.py --evaluate

真正的逻辑都在 fitxray/cli.py 里。
"""

import sys

from fitxray.cli import main

if __name__ == "__main__":
    sys.exit(main())
