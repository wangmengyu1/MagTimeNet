# -*- coding: utf-8 -*-


import os
import itertools
import numpy as np
from sklearn import preprocessing as prep
import tensorflow as tf

'''
Three signal components
'''

# Get signal array from file
def getSigArr(path, sigNorm='minmax'):
    sig = np.load(path)
    return sig  # Directly return sig; no need for expand_dims since this is 3-channel

# Convert segmentation labels to one-hot format
def getSegmentationArr(path, nClasses=2, output_length=1440, class_value=[0,1]):
    seg_labels = np.zeros([output_length, nClasses])
    seg = np.load(path)
    for i in range(nClasses):
        seg_labels[:, i] = (seg == class_value[i]).astype(float)
    return seg_labels

    # The following part is unreachable due to return above,
    # but included below in case of reuse or debug

    seg_labels = np.zeros([output_length, nClasses])
    try:
        seg = np.load(path)
    except Exception as e:
        raise ValueError(f"Error loading file {path}: {e}")

    print(f"seg shape: {seg.shape}, seg dtype: {seg.dtype}")
    if seg.shape == ():  # Check if seg is scalar
        raise ValueError(f"seg is scalar instead of array, file path: {path}")

    # Ensure seg is of integer type
    if not np.issubdtype(seg.dtype, np.integer):
        if np.issubdtype(seg.dtype, np.floating):
            print(f"Warning: seg is of float64 type, converting to int")
            seg = seg.astype(int)
        else:
            raise ValueError(f"Expected integer or boolean type for seg, got {seg.dtype}")

    for i, class_val in enumerate(class_value):
        print(f"class_value[{i}]={class_val}")
        seg_labels[:, i] = (seg == class_val).astype(float)

    return seg_labels

# Data generator for signal-segmentation pairs
def SigSegmentationGenerator(sigs_path, segs_path, batch_size, n_classes, output_length=1440):
    sigs = [s for s in os.listdir(sigs_path) if s.endswith('.npy')]
    segmentations = [s for s in os.listdir(segs_path) if s.endswith('.npy')]

    # Ensure both sigs and segmentations are sorted by filename
    sigs.sort()
    segmentations.sort()

    paired_sigs = []
    paired_segs = []

    for sig in sigs:
        sig_name = os.path.splitext(sig)[0]
        for seg in segmentations:
            seg_name = os.path.splitext(seg)[0]
            if sig_name == seg_name:
                paired_sigs.append(sigs_path + sig)
                paired_segs.append(segs_path + seg)
                break

    assert len(paired_sigs) == len(paired_segs)
    zipped = itertools.cycle(zip(paired_sigs, paired_segs))
    # print("Debug zipped pairs:", zipped)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            sig, seg = next(zipped)
            X.append(getSigArr(sig))  # Directly append signal
            Y.append(getSegmentationArr(seg, n_classes, output_length))
        yield np.array(X), np.array(Y)  # X shape: (batch_size, time_steps, n_features)


