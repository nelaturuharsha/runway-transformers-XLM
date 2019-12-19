#! /usr/bin/env python3
# coding=utf-8

#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions andbased
#limitations under the License.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np
from transformers import XLNetConfig
from transformers import XLNetLMHeadModel, XLNetTokenizer
from runway.data_types import *
import runway

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


MODEL_CLASSES = {
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer)
}
model_class, tokenizer_class = MODEL_CLASSES['xlnet']
tokenizer_class.from_pretrained('xlnet-base-cased')
model_class.from_pretrained('xlnet-base-cased')
tokenizer_class.from_pretrained('xlnet-large-cased')
model_class.from_pretrained('xlnet-large-cased')
