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

from os.path import exists
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt


# ## Question 0

# ### Data Files

# Here is the [link](https://www.eia.gov/consumption/residential/data/2009/csv/recs2009_public.csv) to the 2009 RECS microdata file and here is the [link](https://www.eia.gov/consumption/residential/data/2009/csv/recs2009_public_repweights.csv) to the replicate weights in the 2009 data year. 
#
# Here is the [link](https://www.eia.gov/consumption/residential/data/2015/csv/recs2015_public_v4.csv) to the 2015 RECS microdata file.

# ### Variables

# I will use `HDD65` which reprensents heating degree days in 2009 base temperature 65F and `CDD65` which reprensents cooling degree days in 2009 base temperature 65F.
#
# Also, I will use `DOEID` which reprensents unique identifier for each respondent.
#
# In addition, I will use `REGIONC`, `NWEIGHT`.

# ### Weights and Replicate Weights

# Here is the [link](https://www.eia.gov/consumption/residential/data/2015/pdf/microdata_v3.pdf) that explains how to use the replicate weights.

# To estimate standard errors for weighted point estimates, we first calculate the  estimator's variance as follow
# $$ \hat{V}(\theta)=\frac{1}{R(1-\varepsilon)^{2}} \sum_{r=1}^{R}(\hat{\theta}_{r}-\hat{\theta})^{2},$$
# where $R$ is the number of replicate subsamples, $\varepsilon=0.5$, $\hat{\theta}$ is the full sample based estimator and $\hat{\theta}_{r}$ is estimated from the r-th replicate subsample by using replicate weights. Then
# the standard error of an estimator is
# $$SE = \sqrt{\hat{V}(\theta)}.$$

# #### Reference
# [1] *Residential Energy Consumption Survey (RECS): Using the 2015 microdata file to compute estimates and standard errors (RSEs)*

# ## Question 1

# ### part a)

def creat_type(x):
    """
    This function changes a appropriate format for column.

    Parameters
    ----------
    x : dataframe
        the column needs to be change.

    Returns
    -------
    the changed column.

    """
    if x == 1:
        return "Northeast"
    elif x == 2:
        return "Midwest"
    elif x == 3:
        return "South"
    elif x == 4:
        return "West"


# #### 2009

# +
#read data
base_url = "https://www.eia.gov/consumption/residential/data/"
file_09 = base_url + "2009/csv/recs2009_public.csv"
file_09w = base_url + "2009/csv/recs2009_public_repweights.csv"
col_select = ["DOEID", "HDD65", "CDD65", "REGIONC", "NWEIGHT"]

if not exists("recs2009_public.csv"):
    df_09_a = pd.read_csv(file_09, sep=",", index_col=False, 
                             low_memory=False)
    df_09_a.to_csv("recs2009_public.csv", index=False)
else:
    df_09_a = pd.read_csv("recs2009_public.csv", low_memory=False)

#change columns format
df_09_a = df_09_a[col_select]
df_09_a["REGIONC"] = df_09_a["REGIONC"].map(lambda x: creat_type(x))
df_09_a.rename(columns={"DOEID": "id", "HDD65": "heating", 
                        "CDD65": "cooling", "REGIONC": "region", 
                        "NWEIGHT": "nweight"}, inplace=True)
# -

#construct datasets containing the minimal necessary variables for 2009
df_09_a

# #### 2015

# +
#read data
file_15 = base_url + "2015/csv/recs2015_public_v4.csv"

if not exists("recs2015_public_v4.csv"):
    df_15_a = pd.read_csv(file_15, sep=",")
    df_15_a.to_csv("recs2015_public_v4.csv")
else:
    df_15_a = pd.read_csv("recs2015_public_v4.csv", index_col=0)

#change columns format
df_15_a = df_15_a[col_select]
df_15_a["REGIONC"] = df_15_a["REGIONC"].map(lambda x: creat_type(x))
df_15_a.rename(columns={"DOEID": "id", "HDD65": "heating", 
                        "CDD65": "cooling", "REGIONC": "region", 
                        "NWEIGHT": "nweight"}, inplace=True)
# -

#construct datasets containing the minimal necessary variables for 2015
df_15_a

# ### part b)

# #### 2009

# +
#read data
if not exists("recs2009_public_repweights.csv"):
    df_09_b = pd.read_csv(file_09w, sep=",", index_col=False)
    df_09_b.to_csv("recs2009_public_repweights.csv", index=False)
else:
    df_09_b = pd.read_csv("recs2009_public_repweights.csv")

#change columns format
df_09_b = df_09_b.drop(["NWEIGHT"], axis=1)
df_09_b.rename(columns={"DOEID": "id"}, inplace=True)
for key in df_09_b.columns:
    if key[:10] == "brr_weight":
        l = key[11:len(key)]
        df_09_b.rename(columns={key: "brr_weight{}".format(l)}, inplace=True)

#change to one weight and residence per row 
df1_b = df_09_b.set_index(["id"])
df1_b.columns = [244 * ["brr_weight"], [str(x) for x in range(1, 245)]]
df1_b
df1_b.columns.names = (None, 'time')
df1_b = df1_b.stack()
df1_b.reset_index(inplace=True)
df1_b = df1_b.drop(["time"], axis=1)
# -

df1_b
#construct datasets containing the unique case ids and the replicate weights
#for 2009

# #### 2015

# +
#read data
if not exists("recs2015_public_v4.csv"):
    df_15_b = pd.read_csv(file_15, sep=",")
    df_15_b.to_csv("recs2015_public_v4.csv")
else:
    df_15_b = pd.read_csv("recs2015_public_v4.csv", index_col=0)

#change columns format
col = []
for column in df_15_b.columns:
    if column[:5] == "BRRWT":
        l = column[5:len(column)]
        df_15_b.rename(columns={column: "brr_weight{}".format(l)},
                       inplace=True)
        col.append("brr_weight{}".format(l))
    elif column == "DOEID":
        df_15_b.rename(columns={column: "id"}, inplace=True)
        col.append("id")
df_15_b = df_15_b[col]

#change to one weight and residence per row
df2_b = df_15_b.set_index(["id"])
df2_b.columns = [96 * ["brr_weight"], [str(x) for x in range(1, 97)]]
df2_b
df2_b.columns.names = (None, 'time')
df2_b = df2_b.stack()
df2_b.reset_index(inplace=True)
df2_b = df2_b.drop(["time"], axis=1)
# -

#construct datasets containing the unique case ids and the replicate weights
#for 2015
df2_b


# ## Question 2

# ### part a)

def mean_est(df, index, weight):
    """
    This function estimate the points.

    Parameters
    ----------
    df : dataframe
        the dataframe needs calculation.
    index : str
        the columns in the dataframe needs calculation.
    weight : str
        the weight used in calculation.

    Returns
    -------
    dataframe.

    """
    grouped = df.groupby("region").apply(lambda x: (x[index] * x[weight]).sum() 
                                         / x[weight].sum())
    df_est = grouped.to_frame()
    df_est.columns = [index + " mean"]
    return df_est


def se_est(df, df_mean, index, length):
    """
    This function estimate standard errors.

    Parameters
    ----------
    df : dataframe
        the dataframe needs calculation.
    df_mean : 
        the dataframe contains the mean
    index: str
        the columns in the dataframe needs calculation.
    length: str
        the length of the weights.

    Returns
    -------
    dataframe.

    """
    for i in range(1, length + 1):
        weight = "brr_weight{}".format(i)
        if i == 1:
            df1 = mean_est(df, index, weight)
        else: 
            df2 = mean_est(df, index, weight)
            df1 = pd.concat([df1, df2], axis=1)
    df_temp = deepcopy(df1)
    for i in range(length):
        df_temp.iloc[:, i] = (df1.iloc[:,i] - df_mean[index + " mean"]) ** 2
    grouped = df_temp.apply(lambda x: (x.mean() * 4) ** 0.5, axis=1)
    df_se = grouped.to_frame()
    df_se.columns = ["{} std error".format(index)]
    return df_se


# +
df_09 = pd.concat([df_09_a, df_09_b], axis=1)
df_15 = pd.concat([df_15_a, df_15_b], axis=1)

df1_mean = mean_est(df_09, "heating", "nweight")
df2_mean = mean_est(df_09, "cooling", "nweight")
df3_mean = mean_est(df_15, "heating", "nweight")
df4_mean = mean_est(df_15, "cooling", "nweight")
# -

#the mean of heating and cooling degree days for residences in each Census 
#region for 2009
df_09_mean = pd.concat([df1_mean, df2_mean], axis=1)
df_09_mean

#the mean of heating and cooling degree days for residences in each Census 
#region for 2015
df_15_mean = pd.concat([df3_mean, df4_mean], axis=1)
df_15_mean

df1_se = se_est(df_09, df_09_mean, "heating", 244)
df2_se = se_est(df_09, df_09_mean, "cooling", 244)
df3_se = se_est(df_15, df_15_mean, "heating", 96)
df4_se = se_est(df_15, df_15_mean, "cooling", 96)

#the standard errors  of heating and cooling degree days for residences in each 
#Census region for 2009
df_09_se = pd.concat([df1_se, df2_se], axis=1)
df_09_se

#the standard errors  of heating and cooling degree days for residences in each 
#Census region for 2015
df_15_se = pd.concat([df3_se, df4_se], axis=1)
df_15_se

# +
#construct 95% confidece intervals
df_09_est = pd.concat([df_09_mean, df_09_se], axis=1)
df_15_est = pd.concat([df_15_mean, df_15_se], axis=1)

for x in ["heating", "cooling"]:
    df_09_est[x + " low"] = df_09_est[x + " mean"] - df_09_est[x + " std error"] * 1.96
    df_09_est[x + " high"] = df_09_est[x + " mean"] + df_09_est[x + " std error"] * 1.96
    df_15_est[x + " low"] = df_15_est[x + " mean"] - df_15_est[x + " std error"] * 1.96
    df_15_est[x + " high"] = df_15_est[x + " mean"] + df_15_est[x + " std error"] * 1.96
    
index1 = ["heating"] * 4
index1.extend(["cooling"] * 4)
index2 = ["mean", "std error", "95% CI (low)", "95% CI (high)"]
index3 = ["heating mean", "heating std error", "heating low", "heating high"]
index3.extend(["cooling mean", "cooling std error", "cooling low", "cooling high"])
# -

#95% confidece intervals for 2009
df_09_2a = df_09_est[index3]
df_09_2a.columns = [index1, 2 * index2]
df_09_2a

#95% confidece intervals for 2015
df_15_2a = df_15_est[index3]
df_15_2a.columns = [index1, 2 * index2]
df_15_2a

# ### part b)

df_mean_2b = df_15_mean - df_09_mean
df_se_2b = (df_09_se ** 2 + df_15_se ** 2) ** .5
df_est_2b = pd.concat([df_mean_2b, df_se_2b], axis=1)
for x in ["heating", "cooling"]:
    df_est_2b[x + " low"] = df_est_2b[x + " mean"] - df_est_2b[x + " std error"] * 1.96
    df_est_2b[x + " high"] = df_est_2b[x + " mean"] + df_est_2b[x + " std error"] * 1.96

#the change in heating and cooling degree days between 2009 and 2015 for each Census region
df_2b = df_est_2b[index3]
df_2b.columns = [index1, 2 * index2]
df_2b

# ## Question 3

# ### a)

# +
error_params = dict(elinewidth=2, ecolor='black', capsize=10)

plt.bar(range(4), df_09_mean.iloc[:,0], width=.4, 
        yerr=df_09_se.iloc[:,0], error_kw=error_params,
        label="2009 heating")
plt.bar([i + .4 for i in range(4)], df_15_mean.iloc[:,0], width=.4, 
        yerr=df_15_se.iloc[:,0], error_kw=error_params,
        label="2015 heating")
plt.xticks([i + .2 for i in range(4)], df_09_mean.index)

plt.xlabel("region")
plt.ylabel("average number of heating")
plt.title("The average number of heating degree days")
plt.legend()
plt.show()
# -

# Figure 1: The average number of heating degree days for residences in each Census region for both 2009 and 2015.

# +
plt.bar(range(4), df_09_mean.iloc[:,1], width=.4, 
        yerr=df_09_se.iloc[:,1], error_kw=error_params,
        label="2009 cooling")
plt.bar([i + .4 for i in range(4)], df_15_mean.iloc[:,1], width=.4, 
        yerr=df_15_se.iloc[:,1], error_kw=error_params, 
        label="2015 cooling")
plt.xticks([i + .2 for i in range(4)], df_09_2a.index)

plt.xlabel("region")
plt.ylabel("average number of cooling")
plt.title("The average number of cooling degree days")
plt.ylim(0, 3450)
plt.legend()
plt.show()
# -

# Figure 1: The average number of cooling degree days for residences in each Census region for both 2009 and 2015.

# ### b)

# +
error_params = dict(elinewidth=2, ecolor='black', capsize=10)

plt.bar(range(4), df_mean_2b.iloc[:,0], width=.4, 
        yerr=df_se_2b.iloc[:,0], error_kw=error_params,
        label="change in heating")
plt.bar([i + .4 for i in range(4)], df_mean_2b.iloc[:,1], width=.4, 
        yerr=df_se_2b.iloc[:,1], error_kw=error_params,
        label="change in cooling")
plt.xticks([i + .2 for i in range(4)], df_09_mean.index)

plt.xlabel("region")
plt.ylabel("the change in heating and cooling degree days")
plt.title("The change in heating and cooling degree days")
plt.ylim(-700, 800)
plt.legend()
plt.show()
# -

# Figure 3: The change in heating and cooling degree days between 2009 and 2015 for each Census region.
