"""Configuration constants for TrackNetV3."""

import math

# Image dimensions
HEIGHT = 288
WIDTH = 512

# Gaussian heatmap parameters
SIGMA = 2.5

# Distance threshold
DELTA_T = 1 / math.sqrt(HEIGHT**2 + WIDTH**2)
COOR_TH = DELTA_T * 50

# Image format
IMG_FORMAT = "png"
