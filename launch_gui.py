#!/usr/bin/env python3
"""
Simple launcher for Z-Image Distillation GUI.
Checks dependencies and launches the GUI.
"""

import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    # Check tkinter
    try:
        import tkinter
    except ImportError:
        missing.append("tkinter (usually comes with Python)")
    
    # Check PIL
    try:
        from PIL import Image, ImageTk
    except ImportError:
        missing.append("Pillow (pip install Pillow)")
    
    # Check psutil
    try:
        import psutil
    except ImportError:
        print("Warning: psutil not installed. System monitoring will be limited.")
        print("Install with: pip install psutil")
    
    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall missing dependencies and try again.")
        return False
    
    return True

def main():
    """Launch GUI."""
    print("Z-Image Distillation Trainer GUI")
    print("=" * 40)
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("All dependencies found!")
    print()
    print("Launching GUI...")
    print()
    
    # Launch GUI
    try:
        subprocess.run([sys.executable, "z_image_distillation_gui.py"])
    except KeyboardInterrupt:
        print("\nGUI closed.")
    except Exception as e:
        print(f"\nError launching GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
