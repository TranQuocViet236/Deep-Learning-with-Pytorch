import itertools
import os
import os.path as osp
import random
import xml.etree.ElementTree as ET
from math import sqrt

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn as nn
import torch.utils.data as data

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

