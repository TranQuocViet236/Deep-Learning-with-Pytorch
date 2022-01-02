import os
import os.path as osp

import random
import xml.etree.ElementTree as ET
import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import itertools
from math import sqrt


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

