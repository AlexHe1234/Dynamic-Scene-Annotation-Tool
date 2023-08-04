import sys
import os


home_dir = os.path.join(os.path.dirname(os.path.abspath(__file__))[:-4])
sys.path.append(home_dir)

from config import cfg
