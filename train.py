import os
import numpy as np
import logging
import h5py
from skimage import transform
from skimage import util
from skimage import measure
import cv2
from PIL import Image
import shutil
import png
import itertools
import pydicom # for reading dicom files
import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import math as mt
import matplotlib.pyplot as plt

# load data
data = read_data.load_and_maybe_process_data(
        input_folder=config.data_root,
        preprocessing_folder=config.preprocessing_folder,
        mode=config.data_mode,
        size=config.image_size,
        target_resolution=config.target_resolution,
        force_overwrite=False
    )
