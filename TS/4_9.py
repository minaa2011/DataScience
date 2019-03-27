import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.fftpack
from scipy.signal import butter, lfilter
from scipy.stats import kurtosis

# Plot the yearly temperatures for Norway, Finland, Singapore and Cambodia. Use DTW to measure
# the similarities between the temperature data of these countries and reflect on the results