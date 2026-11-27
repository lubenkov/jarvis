import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from jarvis.chat_format import *


if __name__ == "__main__":
    from jarvis.chat_format import main

    main()
