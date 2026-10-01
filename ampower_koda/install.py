"""Install-time setup: the headless browser Koda's page checks need."""

import os
import subprocess
import sys


def ensure_browser():
    """Download Playwright's Chromium into this bench's environment if it is missing.

    Runs on install and on every migrate; an installed browser makes this a quick
    no-op. It never fails the install or migrate: without a browser, page checks
    report themselves unavailable and the rest of Koda works. A separately
    configured Playwright (KODA_PLAYWRIGHT_PYTHONPATH or KODA_VERIFICATION_LAB) is
    left alone. Linux system libraries need sudo (`playwright install-deps chromium`)
    and cannot be installed from here.
    """
    if os.environ.get("KODA_PLAYWRIGHT_PYTHONPATH") or os.environ.get("KODA_VERIFICATION_LAB"):
        return
    try:
        result = subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"],
                                capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"Koda: could not install Chromium for page checks ({type(exc).__name__}); "
              "run `playwright install chromium` in the bench environment.")
        return
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-500:]
        print(f"Koda: Chromium for page checks was not installed: {detail}")
