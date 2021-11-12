# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### My name: Yumeng Wang
# ### My UM email: yumengwa@umich.edu

# +
import pickle
import random

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import binom, norm
from confidence_interval import ci_mean, ci_prop
# -

# ## Question 0

# ## Working with missing data
#
# ### Detecting
# - use `.isnull()` to detecte missing values
# - use `.notnull()` to detecte non-null values
#
# ### Example

df = pd.DataFrame(
    [[1, np.nan, 3], [np.nan, 5, 6], [7, 8, np.nan], [10, 11, 12]], 
    index=["a", "b", "c", "d"], 
    columns=["0", "1", "2"]
)
df

df.isnull()

# ### Deleting
# - use `.dropna()` to delete the missing values
#
# ### Example

df.dropna()

# ### Filling
# - use `.fillna()` to fill the missing values
# - use `.ffill()` to fill the missing value with forward one
# - use `.bfill()` to fill the missing value with backward one
#
#
# ### Example

df.fillna(100)
