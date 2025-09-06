"""
Utility functions for audio processing and data augmentation
"""

import numpy as np
import random
import torch

def crop_first(data, crop_size=128):
    """첫 번째 부분을 크롭"""
    return data[0: crop_size, :]

def random_crop(data, crop_size=128):
    """랜덤 위치에서 크롭"""
    if data.shape[0] <= crop_size:
        return data
    start = int(random.random() * (data.shape[0] - crop_size))
    return data[start: (start + crop_size), :]

def random_mask(data, rate_start=0.1, rate_seq=0.2):
    """랜덤하게 일부 구간을 마스킹"""
    new_data = data.copy()
    mean = new_data.mean()
    prev_zero = False
    for i in range(new_data.shape[0]):
        if random.random() < rate_start or (prev_zero and random.random() < rate_seq):
            prev_zero = True
            new_data[i, :] = mean
        else:
            prev_zero = False
    return new_data

def random_multiply(data):
    """랜덤한 배율로 곱하기"""
    new_data = data.copy()
    return new_data * (0.9 + random.random() / 5.)
