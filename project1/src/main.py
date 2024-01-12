import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.system import System

if __name__ == "__main__":
    system = System("parameters.conf")
    system.run()
