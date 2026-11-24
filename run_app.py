"""
AI-Based Motorcycle Helmet Detection and Rider Safety Monitoring System
Direct Python Launcher
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def main():
    print("=" * 75)
    print("  AI-Based Motorcycle Helmet Detection and Rider Safety Monitoring System")
    print("  Launching Streamlit Server...")
    print("=" * 75)

    # Launch streamlit via subprocess
    cmd = [sys.executable, "-m", "streamlit", "run", str(BASE_DIR / "app.py")]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[+] Application stopped by user.")

if __name__ == "__main__":
    main()
