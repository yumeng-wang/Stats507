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

from IPython.display import display
import time
import numpy as np
import pandas as pd


# ## Question 0  Code review warmup

# ### Code snippet

# + active=""
# sample_list = [(1, 3, 5), (0, 1, 2), (1, 9, 8)]
# op = []
# for m in range(len(sample_list)):
#     li = [sample_list[m]]
#         for n in range(len(sample_list)):
#             if (sample_list[m][0] == sample_list[n][0] and
#                     sample_list[m][3] != sample_list[n][3]):
#                 li.append(sample_list[n])
#         op.append(sorted(li, key=lambda dd: dd[3], reverse=True)[0])
# res = list(set(op))
# -

# ### Code Reviews

# #### a.  
# For the tuples in the list with the same first element and different last elements, this function will collect the tuple having the largest last element. Then this function will return a list containing the tuples with the different first elements.

# #### b.
# 1. Please iterate over values instead of iterating over indices. For example, using `for x in sample_list:` help the code look more concise.
# 2. Please use the correct indentation. For example, there is no need to use another indentation in the fifth line before a new `for` loop.
# 3. Please avoid using the specific number in the code. For example, using `len(sample_list[m]) - 1` replace `3` help the code more applicable. 

# ## Question 1 List of Tuples

def generate_list(n, k=3, low=1, high=10):
    """
    This function  generate a random list of n k-tuples containing integers ranging from low to high.

    Parameters
    ----------
    n : int
        the length of list.
    k : int, optional
        the length of tuple. The default is 3.
    low : int, optional
        The range of tuple. The default is 1.
    high : int, optional
        The range of tuple. The default is 10.

    Returns
    -------
    List of n k-tuples.

    """
    li = []
    for i in range(n):
        x = np.sort(np.random.randint(low, high, k))
        li.append(tuple(x))
    return li


assert type(generate_list(5)) is list
assert type(generate_list(5)[0]) is tuple


# ## Question 2 Refactor the Snippet

# #### a.  

def select_a(sample_list, id1=0, id2=2):
    """
    This function collect the tuple having the largest element in the specific index 
    with the sample element in the specific index.

    Parameters
    ----------
    sample_list : list
        list of n k-tuples.
    idx1 : int, optional
        indict the index to choose the tuple having the largest element. The default 
        is 0.
    idx2 : int, optional
        indict the index to choose the tuple having the sample element. The default 
        is 2.

    Returns
    -------
    List.

    """
    op = []
    for m in range(len(sample_list)):
        li = [sample_list[m]]
        for n in range(len(sample_list)):
            if (sample_list[m][id1] == sample_list[n][id1] and
                    sample_list[m][id2] != sample_list[n][id2]):
                li.append(sample_list[n])
        op.append(sorted(li, key=lambda dd: dd[id2], reverse=True)[0])
    res = list(set(op))
    return res


# #### b.  

def select_b(sample_list, id1=0, id2=2):
    """
    This function collect the tuple having the largest element in the specific index 
    with the sample element in the specific index.

    Parameters
    ----------
    sample_list : list
        list of n k-tuples.
    idx1 : int, optional
        indict the index to choose the tuple having the largest element. The default 
        is 0.
    idx2 : int, optional
        indict the index to choose the tuple having the sample element. The default 
        is 2.

    Returns
    -------
    List.

    """
    op = []
    for x in sample_list:
        li = [x]
        for y in sample_list:
            if (x[id1] == y[id1] and x[id2] != y[id2]):
                li.append(y)
        op.append(sorted(li, key=lambda dd: dd[id2], reverse=True)[0])
    res = list(set(op))
    return res


# #### c.  

def select_c(sample_list, id1=0, id2=2):
    """
    This function collect the tuple having the largest element in the specific index 
    with the sample element in the specific index.

    Parameters
    ----------
    sample_list : list
        list of n k-tuples.
    idx1 : int, optional
        indict the index to choose the tuple having the largest element. The default 
        is 0.
    idx2 : int, optional
        indict the index to choose the tuple having the sample element. The default 
        is 2.

    Returns
    -------
    List.

    """
    res = []
    dict = {}
    for x in sample_list:
        if x[id1] not in dict.keys():
            dict[x[id1]] = [x]
        else:
            if x[id2] > dict[x[id1]][0][id2]:
                dict[x[id1]] = [x]
            elif x[id2] == dict[x[id1]][0][id2]:
                dict[x[id1]].append(x) 
    for value in dict.values():
        res.extend(value)
    return res


# #### d.  

def calculate_time(func, sample_list, rep=20):
    """
    This function calculates running time of each functions by Monte Carlo study. 

    Parameters
    ----------
    func : function
        the function name.
    sample_list : list
        list of n k-tuples.
    rep : int, optional
        repeated number of calculating the running time. The default is 20.

    Returns
    -------
    the median running time of each functions.

    """
    cal_time = []
    for i in range(rep):
        start = time.perf_counter()
        func(sample_list)
        end = time.perf_counter()
        cal_time.append(end - start)
    return round(np.mean(cal_time), 4)


test_list = generate_list(600, 5)
time_a = calculate_time(select_a, test_list)
time_b = calculate_time(select_b, test_list)
time_c = calculate_time(select_c, test_list)
run_time = {"method a": time_a, "method b": time_b, "method c": time_c}
df_time = pd.DataFrame(run_time, index=["running time"])
display(df_time) # the execution times of the three functions above 

# ## Question 3

# #### a.

# +
col = ["SEQN", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "DMDMARTL", 
       "RIDSTATR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTINT2YR"]

df_DEMO_G = pd.read_sas("DEMO_G.XPT")[col]
df_DEMO_G["cohort"] = "2011-2012"

df_DEMO_H = pd.read_sas("DEMO_H.XPT")[col]
df_DEMO_H["cohort"] = "2013-2014"

df_DEMO_I = pd.read_sas("DEMO_I.XPT")[col]
df_DEMO_I["cohort"] = "2015-2016"

df_DEMO_J = pd.read_sas("DEMO_J.XPT")[col]
df_DEMO_J["cohort"] = "2017-2018"

df_DEMO = pd.concat([df_DEMO_G, df_DEMO_H, df_DEMO_I, df_DEMO_J])
df_DEMO.columns = ["id", "age", "race", "education", "marriage", "weight1", 
                   "weight2", "weight3", "weight4", "weight5", "year"]
df_DEMO.index = range(1, df_DEMO.shape[0] + 1)
df_DEMO.to_pickle('DEMO.pkl')
df_DEMO

# +
col = ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "DMDMARTL", 
       "RIDSTATR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTINT2YR"]

df_DEMO_G = pd.read_sas("DEMO_G.XPT")[col]
df_DEMO_G["cohort"] = "2011-2012"

df_DEMO_H = pd.read_sas("DEMO_H.XPT")[col]
df_DEMO_H["cohort"] = "2013-2014"

df_DEMO_I = pd.read_sas("DEMO_I.XPT")[col]
df_DEMO_I["cohort"] = "2015-2016"

df_DEMO_J = pd.read_sas("DEMO_J.XPT")[col]
df_DEMO_J["cohort"] = "2017-2018"

df_DEMO = pd.concat([df_DEMO_G, df_DEMO_H, df_DEMO_I, df_DEMO_J])
df_DEMO.columns = ["id", "gender", "age", "race", "education", "marriage", "exam_status", 
                   "weight2", "weight3", "weight4", "weight5", "year"]
df_DEMO.index = range(1, df_DEMO.shape[0] + 1)
df_DEMO.to_pickle('DEMO.pkl')
df_DEMO
# -

# #### b.

# +
col = ["SEQN", "OHDDESTS"]
df_columns = pd.read_sas("OHXDEN_G.XPT").columns
for column in df_columns:
    if column[:3] == "OHX" and column[-2:] == "TC":
        col.append(column)
        
df_OHXDEN_G = pd.read_sas("OHXDEN_G.XPT")[col]
df_OHXDEN_G["cohort"] = "2011-2012"

df_OHXDEN_H = pd.read_sas("OHXDEN_H.XPT")[col]
df_OHXDEN_H["cohort"] = "2013-2014"

df_OHXDEN_I = pd.read_sas("OHXDEN_I.XPT")[col]
df_OHXDEN_I["cohort"] = "2015-2016"

df_OHXDEN_J = pd.read_sas("OHXDEN_J.XPT")[col]
df_OHXDEN_J["cohort"] = "2017-2018"

df_OHXDEN = pd.concat([df_OHXDEN_G, df_OHXDEN_H, df_OHXDEN_I, df_OHXDEN_J])
df_OHXDE_col = ["id", "ohx_status"]
df_OHXDE_col.extend(["tooth_count{}".format(x) for x in range(1,33)])
df_OHXDE_col.extend(["coronal_caries{}".format(x) for x in range(2,16)])
df_OHXDE_col.extend(["coronal_caries{}".format(x) for x in range(18,32)])
df_OHXDE_col.append("year")
df_OHXDEN.columns = df_OHXDE_col
df_OHXDEN.index = range(1, df_OHXDEN.shape[0] + 1)
df_OHXDEN.to_pickle('OHXDEN.pkl')
df_OHXDEN
# -

# #### c.

df_DEMO.shape[0] # the number of cases in a.

df_OHXDEN.shape[0] # the number of cases in .
