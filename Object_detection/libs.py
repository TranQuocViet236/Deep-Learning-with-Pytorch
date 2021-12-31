import os
import os.path as osp

import random
import xml.etree.ElementTree as ET
import cv2
import torch
import torch.utils.data as data
import numpy as np

import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

