#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
from typing import Any, List, Set, Dict, Tuple, Optional, Iterable, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.convolutional import GatedGraphConvolution

warnings.simplefilter("ignore")

