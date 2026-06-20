# Copyright (c) Twinkle Contributors. All rights reserved.
"""Allow running as `python -m twinkle_client.tui`."""

import sys

from twinkle_client.tui import main

if __name__ == '__main__':
    sys.exit(main())
