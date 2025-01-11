from collections import OrderedDict
from typing import Tuple, Union
from einops import rearrange
from sparselinear import SparseLinear
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import cbm
import torch
from tqdm import tqdm

from timm.utils import accuracy
import os
import random
import cbm_utils
import data_utils
import similarity
import argparse
import datetime
import torch.optim as optim
import json
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset
import video_utils
import torch.distributed as dist
import torch
import numpy as np
from PIL import Image
from IPython.display import display, Image as IPImage

def debug(args,save_name):
    d_val = args.data_set + "_test"
    val_data_t = data_utils.get_data(d_val,args=args)
    val_data_t.end_point = 2
    device = torch.device(args.device)

    model,_ = cbm.load_cbm_two_stream(save_name, device, args)
           
    print("?***? Start test")
    accuracy = cbm_utils.get_accuracy_cbm(model, val_data_t, device , 64, 10)
    # counted_object,counted_action,counted_scene = concept_counts
    # 전체 테스트 샘플 개수 (total_samples)를 직접 넣어주세요.
    total_samples = len(val_data_t)
    print("=== Basic Information ===")
    print("?****? Accuracy: {:.2f}%".format(accuracy*100))
    print("=========================\n\n")