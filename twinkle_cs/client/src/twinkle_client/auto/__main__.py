# Copyright (c) Twinkle Contributors. All rights reserved.
"""Allow running as `python -m twinkle_client.auto`."""

import sys

from twinkle_client.auto import main

if __name__ == '__main__':
    sys.exit(main())
