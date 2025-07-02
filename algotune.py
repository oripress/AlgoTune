#!/usr/bin/env python3

# Simple launcher that delegates to scripts/algotune.py
# This allows users to run python3 algotune.py from project root

import subprocess
import sys
from pathlib import Path

script_dir = Path(__file__).parent
script_path = script_dir / "scripts/algotune.py"

# Run the actual script with all arguments
result = subprocess.run([sys.executable, str(script_path)] + sys.argv[1:])
sys.exit(result.returncode)