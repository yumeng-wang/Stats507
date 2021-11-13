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

# +
import pickle
import random
import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from os.path import exists

random.seed(2021)
# -

# ## Question 0

# +
# tooth growth data
file = 'tooth_growth.feather'
if exists(file):
    tg_data = pd.read_feather(file)
else: 
    tooth_growth = sm.datasets.get_rdataset('ToothGrowth')
    tg_data = tooth_growth.data
    tg_data.to_feather(file)

# transform data
tg_data['log_len'] = tg_data[['len']].transform(np.log)
tg_data['dose_cat'] = pd.Categorical(tg_data['dose'])
tg_data['OJ'] = pd.get_dummies(tg_data['supp'])['OJ']

# fit the model
mod1 = smf.ols('log_len ~ OJ*dose_cat', data=tg_data)
fit1 = mod1.fit()
fit1.summary2()
# -

# The R-Squared is calculated as follow,
# $$R ^ 2 = \frac{\sum_i (\hat y_i - \bar y) ^ 2}{\sum_i (y_i - \bar y) ^ 2}$$

R2 = fit1.ess / fit1.centered_tss
R2

# The Adjusted R-Squared  is calculated as follow,
# $$R ^ 2 \text{-adjusted} = 1 - \frac{(1 - R ^ 2)(n - 1)}{n - p - 1}$$

R2_adj = 1 - (1 - R2) * (fit1.nobs - 1) / (fit1.nobs - 5 - 1)
R2_adj

print(round(R2, 3) == round(fit1.rsquared, 3))
print(round(R2_adj, 3) == round(fit1.rsquared_adj, 3))

# The Computed R sqaure and Adjusted R-Squared  is the same with result object.

# ## Question 1 

# ### a.

# +
# the NHANES dentition and demographics data from problem sets 2 and 4
df_demo = pickle.load(open("DEMO.pkl", "rb"))
df_ohxden = pickle.load(open("OHXDEN.pkl", "rb"))
df_raw = pd.merge(df_demo[["id", "age"]], df_ohxden[df_ohxden.columns[:34]], 
                  how="outer", on=["id"])

# limit the analyses to those age 12 and older
df = df_raw.copy()
df = df[df['age'] >= 12]
df.index = range(len(df))

# transform data to bernoulli variables
for i in range(3, 35):
    df[df.columns[i]] = df[df.columns[i]].apply(lambda x: 1 if x == 2 else 0)
# -

aic = 10000
for k in range(3, 6):
    for i in range(50):
        knot = sorted(random.sample(range(14, 80), k))
        str_bs = 'tooth_count1 ~ bs(age, knots=' + str(knot) + ', degree=3)'
        fit_bs = smf.logit(str_bs, data=df).fit(disp=False)
        aic_new = fit_bs.aic
        if aic_new < aic:
            knot = knot
            aic = aic_new
knot

# To find the best fit model, I tried different lengths of knots and different knots. The procedure is:
# 1. choose the length of knots (3 or 4 or 5),
# 2. randomly choose the values of knots,
# 3. fit the logistic regression model and calculate the AIC value,
# 4. repeat the 1 - 3 for a different length of knots,
# 5. select the model with the minimum AIC value.
#
# Then the final model's knots is [14, 17, 20, 24, 71] and degree is 3

# ### b.

# fit the model to all other teeth in the data
for i in range(3, 35):
    str_bs = str(df.columns[i]) + ' ~ bs(age, knots=' + str(knot) + ', degree=3)'
    fit_bs = smf.logit(str_bs, data=df).fit(disp=False)
    df[str(df.columns[i]) + '_pred'] = fit_bs.predict()

# ### c.

# +
position = (
    list(range(1, 9)) + 
    list(reversed(range(9, 17))) + 
    list(range(17, 25)) + 
    list(reversed(range(25, 33)))
)

# visualization showing the predicted probability for each tooth
df_1c = df.sort_values('age')
fig, ax = plt.subplots(nrows=8, ncols=4, sharex=True, sharey=True)
fig.set_size_inches(16, 28)
for i in range(32):
    r = (position[i] - 1) % 8
    c = i // 8
    str_col = str(df_1c.columns[i + 35])
    ax[r, c].plot(df_1c['age'], df_1c[str_col])
    ax[r, c].set_title('tooth ' + str(i + 1))
    ax[r, c].set_xlabel('age')
    if c == 0:
        ax[r, c].set_ylabel('probability')


# -

# ## Question 2

# ### 1.

# +
def split(values, li):
    cata = []
    for x in values:
        for i in range(10):
            if li[i] <= x and x < li[i+1]:
                cata.append(i)
    return cata

df_2a = df.sort_values('tooth_count1_pred')
df_2a.index = range(len(df_2a.index))

li1 = np.linspace(0, len(df), 11, dtype=int)
li1[10] = li1[10] + 1
df_2a['group'] = split(df_2a.index, li1)
# -

# ### 2.

df_decile = df_2a.groupby('group').mean()[['tooth_count1', 'tooth_count1_pred']]
df_decile.columns = ['observed probabilities', 'expected probabilities']
df_decile

# ### 3.

plt.scatter(df_decile['observed probabilities'], df_decile['expected probabilities'])
plt.plot([0, 0.35], [0, 0.35])
plt.xlabel('observed probabilities')
plt.ylabel('expected probabilities')
plt.show()

# ### 4.

# From the figure, I thinl the model is well-calibrated because the points in this plot fall approximately on this line.
