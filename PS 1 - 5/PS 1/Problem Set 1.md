---
jupyter:
  jupytext:
    encoding: '# -*- coding: utf-8 -*-'
    formats: ipynb,py:light,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import time
import numpy as np
from scipy.stats import norm, beta
```

## Question 0 - Markdown warmup


This is *question 0* for [problem set 1](https://jbhender.github.io/Stats507/F21/ps/ps1.html) of [Stats 507](https://jbhender.github.io/Stats507/F21/).

> Question 0 is about Markdown.

The next question is about the **Fibonnaci sequence**, $F_n=F_{n−2}+F_{n−1}$. In part $\bf{a}$ we will define a Python function `fib_rec()`.

Below is a …

### Level 3 Header

Next, we can make a bulleted list:
- Item 1
    - detail 1
    - detail 2
- Item 2

Finally, we can make an enumerated list:
1. Item 1
2. Item 2
3. Item 3


## Question 1 - Fibonnaci Sequence


#### a.

```python
def fib_rec(n, a = 0, b = 1):
    """
    This function generates Fibonacci numbers by using recursive method.

    Parameters
    ----------
    n : int
        the nth Fibonacci numbers will be calculated.
    a : int, optional
        the first Fibonacci numbers F_0. The default is 0.
    b : int, optional
        the second Fibonacci numbers F_1. The default is 1.

    Returns
    -------
    the nth Fibonacci numbers.

    """
    if n == 0:
        return a
    elif n == 1:
        return b
    else:
        return fib_rec(n-1) + fib_rec(n-2)
```

```python
print(fib_rec(7))
print(fib_rec(11))
print(fib_rec(13))
```

#### b.

```python
def fib_for(n, a = 0, b = 1):
    """
    This function generates Fibonacci numbers by using for loop.

    Parameters
    ----------
    n : int
        the nth Fibonacci numbers will be calculated.
    a : int, optional
        the first Fibonacci numbers F_0. The default is 0.
    b : int, optional
        the second Fibonacci numbers F_1. The default is 1.

    Returns
    -------
    the nth Fibonacci numbers.

    """
    x = a
    y = b
    for k in range(n):
        z = x + y
        x = y
        y = z
    return x
```

```python
print(fib_for(7))
print(fib_for(11))
print(fib_for(13))
```

#### c.

```python
def fib_whl(n, a = 0, b = 1):
    """
    This function generates Fibonacci numbers by using while loop.

    Parameters
    ----------
    n : int
        the nth Fibonacci numbers will be calculated.
    a : int, optional
        the first Fibonacci numbers F_0. The default is 0.
    b : int, optional
        the second Fibonacci numbers F_1. The default is 1.

    Returns
    -------
    the nth Fibonacci numbers.

    """
    x = a
    y = b
    k = 0
    while k < n:
        z = x + y
        x = y
        y = z
        k += 1
    return x
```

```python
print(fib_whl(7))
print(fib_whl(11))
print(fib_whl(13))
```

#### d.

```python
def fib_rnd(n):
    """
    This function generates Fibonacci numbers by using the rounding method.

    Parameters
    ----------
    n : int
        the nth Fibonacci numbers will be calculated.

    Returns
    -------
    the nth Fibonacci numbers.

    """
    phi = (1 + 5 ** .5) / 2
    return round(phi ** n / 5 ** .5)
```

```python
print(fib_rnd(7))
print(fib_rnd(11))
print(fib_rnd(13))
```

#### e.

```python
def fib_flr(n, a = 0, b = 1):
    """
    This function generates Fibonacci numbers by using the truncation method.

    Parameters
    ----------
    n : int
        the nth Fibonacci numbers will be calculated.
in
    Returns
    -------
    the nth Fibonacci numbers.

    """
    phi = (1 + 5 ** .5) / 2
    return int(phi ** n / 5 ** .5 + .5)
```

```python
print(fib_flr(7))
print(fib_flr(11))
print(fib_flr(13))
```

#### f.

```python
def calculate_time(func, n, r):
    """
    This function calculates the median running time of each functions. 

    Parameters
    ----------
    func : function
        the function name.
    n : int
        the nth Fibonacci numbers will be calculated.
    r : int
        repeated number of calculating the running time.

     Returns
     -------
     the median running time of each functions.

    """
    cal_time = []
    for i in range(r):
        start = time.perf_counter()
        func(n)
        end = time.perf_counter()
        cal_time.append(end - start)
    return round(np.mean(cal_time), 4)
```

<!-- #region variables={"calculate_time(fib_rec, 15, 10)": "0.0002", "calculate_time(fib_for, 15, 10)": "0.0", "calculate_time(fib_whl, 15, 10)": "0.0", "calculate_time(fib_rnd, 15, 10)": "0.0", "calculate_time(fib_flr, 15, 10)": "0.0", "calculate_time(fib_rec, 20, 10)": "0.0024", "calculate_time(fib_for, 20, 10)": "0.0", "calculate_time(fib_whl, 20, 10)": "0.0", "calculate_time(fib_rnd, 20, 10)": "0.0", "calculate_time(fib_flr, 20, 10)": "0.0", "calculate_time(fib_rec, 25, 10)": "0.0261", "calculate_time(fib_for, 25, 10)": "0.0", "calculate_time(fib_whl, 25, 10)": "0.0", "calculate_time(fib_rnd, 25, 10)": "0.0", "calculate_time(fib_flr, 25, 10)": "0.0", "calculate_time(fib_rec, 30, 10)": "0.2826", "calculate_time(fib_for, 30, 10)": "0.0", "calculate_time(fib_whl, 30, 10)": "0.0", "calculate_time(fib_rnd, 30, 10)": "0.0", "calculate_time(fib_flr, 30, 10)": "0.0", "calculate_time(fib_rec, 35, 10)": "3.1392", "calculate_time(fib_for, 35, 10)": "0.0", "calculate_time(fib_whl, 35, 10)": "0.0", "calculate_time(fib_rnd, 35, 10)": "0.0", "calculate_time(fib_flr, 35, 10)": "0.0"} -->
| n   | fib_rec | fib_for | fib_whl | fib_rnd | fib_flr |
| :-: | :-----: | :-----: | :-----: | :-----: | :-----: |
|  15 |{{calculate_time(fib_rec, 15, 10)}}|{{calculate_time(fib_for, 15, 10)}}|{{calculate_time(fib_whl, 15, 10)}}|{{calculate_time(fib_rnd, 15, 10)}}|{{calculate_time(fib_flr, 15, 10)}}|
|  20 |{{calculate_time(fib_rec, 20, 10)}}|{{calculate_time(fib_for, 20, 10)}}|{{calculate_time(fib_whl, 20, 10)}}|{{calculate_time(fib_rnd, 20, 10)}}|{{calculate_time(fib_flr, 20, 10)}}|
|  25 |{{calculate_time(fib_rec, 25, 10)}}|{{calculate_time(fib_for, 25, 10)}}|{{calculate_time(fib_whl, 25, 10)}}|{{calculate_time(fib_rnd, 25, 10)}}|{{calculate_time(fib_flr, 25, 10)}}|
|  30 |{{calculate_time(fib_rec, 30, 10)}}|{{calculate_time(fib_for, 30, 10)}}|{{calculate_time(fib_whl, 30, 10)}}|{{calculate_time(fib_rnd, 30, 10)}}|{{calculate_time(fib_flr, 30, 10)}}|
|  35 |{{calculate_time(fib_rec, 35, 10)}}|{{calculate_time(fib_for, 35, 10)}}|{{calculate_time(fib_whl, 35, 10)}}|{{calculate_time(fib_rnd, 35, 10)}}|{{calculate_time(fib_flr, 35, 10)}}|
<!-- #endregion -->

## Question 2 - Pascal’s Triangle


#### a.

```python
def pascal_row(n):
    """
    This function compute the nth row of Pascal’s triangle.

    Parameters
    ----------
    n : int
        compute the nth row of Pascal’s triangle.

    Returns
    -------
    the nth row of Pascal’s triangle.

    """
    x = [1]
    for i in range(n - 1):
        c_i = int(x[i] * (n - i - 1) / (i + 1))
        x.append(c_i)
    return x
```

#### b.

```python
def print_pascal_trl(n):
    """
    This function prints the first n rows of Pascal’s triangle.

    Parameters
    ----------
    n : int
        print the first n rows of Pascal’s triangle. 

    Returns
    -------
    None.

    """
    largest_elm = pascal_row(n)[n // 2]
    elm_width = len(str(largest_elm)) + 1
    trl_width = elm_width * n
    
    for i in range(n):
        x = pascal_row(i + 1)
        y = "".join([str(x[j]).center(elm_width) for j in range(i + 1)])
        print(y.center(trl_width))
```

```python

```

```python
print_pascal_trl(11)
```

## Question 3 - Statistics 101


#### a.

```python
def estimate_ci(data, level, output = "string"):
    """
    This function estimates a point and interval based on Normal theory. 

    Parameters
    ----------
    data : array
        input data.
    level : int
        confidence interval.
    output : string

    Returns
    -------
    string or dictionary.

    """
    mean = np.mean(data)
    std = np.std(data)
    z = norm.ppf(1 - (1 - level * .01) / 2) / (len(data) ** .5)
    est = round(mean, 3)
    lwr = round(mean - z * std, 3)
    upr = round(mean + z * std, 3)
    
    if output == "string":
        return str(est) + '[' + str(level) + '%CI:(' + str(lwr) + ',' + str(upr) + ')]'
    elif output == None:
        return {'est': est, 'lwr': lwr, 'upr': upr}
```

#### b.

```python
def binomial_ci(data, method, level, output = "string"):
    """
    This function estimates a point and interval using four different methods. 

    Parameters
    ----------
    data : array
        input data.
    method : string
        indicate which method will be used. 
        "normal_appx" means normal approximation will be used. 
        "clopper_pearson" means Clopper-Pearson interval will be computed. 
        "Jeffrey" means Jeffrey’s interval interval will be computed. 
        "Agresti-Coull" means Agresti-Coull interval will be computed. 
    level : int
        confidence interval.
    output : string

    Returns
    -------
    string or dictionary.

    """
    x = np.sum(data)
    n = len(data)
    p_hat = x / n
    alpha = 1 - level * .01
    
    if method == "normal_appx":
        est = round(p_hat, 3)
        if np.min([n * p_hat, n * (1 - p_hat)]) > 12:
            z = norm.ppf(1 - alpha / 2)
            lwr = round(p_hat - z * ((p_hat * (1 - p_hat) / n) ** .5), 3)
            upr = round(p_hat + z * ((p_hat * (1 - p_hat) / n) ** .5), 3)
        else:
            print("The condition is not satisfied.")
            return
            
    if method == "clopper_pearson":
        est = round(p_hat, 3)
        lwr = round(beta.ppf(alpha / 2, x, n - x + 1), 3)
        upr = round(beta.ppf(1 - alpha / 2, x + 1, n - x), 3)
        
    if method == "Jeffrey":
        est = round(p_hat, 3)
        lwr = round(np.max([0, beta.ppf(alpha / 2, x + 0.5, n - x + 0.5)]), 3)
        upr = round(np.min([1, beta.ppf(1 - alpha / 2, x + 0.5, n - x + 0.5)]), 3)
        
    if method == "Agresti_Coull":
        z = norm.ppf(1 - alpha / 2)
        n_tilde = n + z ** 2
        p_tilde = (x + z ** 2 / 2) / n_tilde
        est = round(p_tilde, 3)
        lwr = round(p_tilde - z * ((p_tilde * (1 - p_tilde) / n) ** .5), 3)
        upr = round(p_tilde + z * ((p_tilde * (1 - p_tilde) / n) ** .5), 3)
        
    if output == "string":
        return str(est) + '[' + str(level) + '%CI:(' + str(lwr) + ',' + str(upr) + ')]'
    elif output == None:
        return {'est': est, 'lwr': lwr, 'upr': upr}
```

#### c.

```python
data = np.append(np.repeat(1, 42), np.repeat(0, 48)) 
```

<!-- #region variables={"estimate_ci(data, 90)": "0.467[90%CI:(0.38,0.553)]", "binomial_ci(data, \"normal_appx\", 90)": "0.467[90%CI:(0.38,0.553)]", "binomial_ci(data, \"clopper_pearson\", 90)": "0.467[90%CI:(0.376,0.559)]", "binomial_ci(data, \"Jeffrey\", 90)": "0.467[90%CI:(0.382,0.553)]", "binomial_ci(data, \"Agresti_Coull\", 90)": "0.468[90%CI:(0.381,0.554)]", "estimate_ci(data, 95)": "0.467[95%CI:(0.364,0.57)]", "binomial_ci(data, \"normal_appx\", 95)": "0.467[95%CI:(0.364,0.57)]", "binomial_ci(data, \"clopper_pearson\", 95)": "0.467[95%CI:(0.361,0.575)]", "binomial_ci(data, \"Jeffrey\", 95)": "0.467[95%CI:(0.366,0.569)]", "binomial_ci(data, \"Agresti_Coull\", 95)": "0.468[95%CI:(0.365,0.571)]", "estimate_ci(data, 99)": "0.467[99%CI:(0.331,0.602)]", "binomial_ci(data, \"normal_appx\", 99)": "0.467[99%CI:(0.331,0.602)]", "binomial_ci(data, \"clopper_pearson\", 99)": "0.467[99%CI:(0.331,0.606)]", "binomial_ci(data, \"Jeffrey\", 99)": "0.467[99%CI:(0.336,0.601)]", "binomial_ci(data, \"Agresti_Coull\", 99)": "0.469[99%CI:(0.333,0.604)]"} -->
| confidence level | standard                  |  normal approximation                    | clopper-pearson                              | Jeffrey's                              | Agresti-Coull                              |
| :--------------: | :-----------------------: | :--------------------------------------: | :------------------------------------------: | :------------------------------------: | :------------------------------------------: |
| 90\%             | {{estimate_ci(data, 90)}} | {{binomial_ci(data, "normal_appx", 90)}} | {{binomial_ci(data, "clopper_pearson", 90)}} | {{binomial_ci(data, "Jeffrey", 90)}} | {{binomial_ci(data, "Agresti_Coull", 90)}} |
| 95\%               | {{estimate_ci(data, 95)}} | {{binomial_ci(data, "normal_appx", 95)}} | {{binomial_ci(data, "clopper_pearson", 95)}} | {{binomial_ci(data, "Jeffrey", 95)}} | {{binomial_ci(data, "Agresti_Coull", 95)}} |
| 99\%                | {{estimate_ci(data, 99)}} | {{binomial_ci(data, "normal_appx", 99)}} | {{binomial_ci(data, "clopper_pearson", 99)}} | {{binomial_ci(data, "Jeffrey", 99)}} | {{binomial_ci(data, "Agresti_Coull", 99)}} |
<!-- #endregion -->

The length of the different methods.

<!-- #region variables={"round(estimate_ci(data, 90, None)['upr'] - estimate_ci(data, 90, None)['lwr'], 3)": "0.173", "round(binomial_ci(data, \"normal_appx\", 90, None)['upr'] - binomial_ci(data, \"normal_appx\", 90, None)['lwr'], 3)": "0.173", "round(binomial_ci(data, \"clopper_pearson\", 90, None)['upr'] - binomial_ci(data, \"clopper_pearson\", 90, None)['lwr'], 3)": "0.183", "round(binomial_ci(data, \"Jeffrey\", 90, None)['upr'] - binomial_ci(data, \"Jeffrey\", 90, None)['lwr'], 3)": "0.171", "round(binomial_ci(data, \"Agresti_Coull\", 90, None)['upr'] - binomial_ci(data, \"Agresti_Coull\", 90, None)['lwr'], 3)": "0.173", "round(estimate_ci(data, 95, None)['upr'] - estimate_ci(data, 95, None)['lwr'], 3)": "0.206", "round(binomial_ci(data, \"normal_appx\", 95, None)['upr'] - binomial_ci(data, \"normal_appx\", 95, None)['lwr'], 3)": "0.206", "round(binomial_ci(data, \"clopper_pearson\", 95, None)['upr'] - binomial_ci(data, \"clopper_pearson\", 95, None)['lwr'], 3)": "0.214", "round(binomial_ci(data, \"Jeffrey\", 95, None)['upr'] - binomial_ci(data, \"Jeffrey\", 95, None)['lwr'], 3)": "0.203", "round(binomial_ci(data, \"Agresti_Coull\", 95, None)['upr'] - binomial_ci(data, \"Agresti_Coull\", 95, None)['lwr'], 3)": "0.206", "round(estimate_ci(data, 99, None)['upr'] - estimate_ci(data, 99, None)['lwr'], 3)": "0.271", "round(binomial_ci(data, \"normal_appx\", 99, None)['upr'] - binomial_ci(data, \"normal_appx\", 99, None)['lwr'], 3)": "0.271", "round(binomial_ci(data, \"clopper_pearson\", 99, None)['upr'] - binomial_ci(data, \"clopper_pearson\", 99, None)['lwr'], 3)": "0.275", "round(binomial_ci(data, \"Jeffrey\", 99, None)['upr'] - binomial_ci(data, \"Jeffrey\", 99, None)['lwr'], 3)": "0.265", "round(binomial_ci(data, \"Agresti_Coull\", 99, None)['upr'] - binomial_ci(data, \"Agresti_Coull\", 99, None)['lwr'], 3)": "0.271"} -->
| confidence level | standard |  normal approximation | clopper-pearson | Jeffrey's | Agresti-Coull  |
| :--------------: | :------: | :-------------------: | :-------------: | :-------: | :------------: |
| 90\% | {{round(estimate_ci(data, 90, None)['upr'] - estimate_ci(data, 90, None)['lwr'], 3)}} | {{round(binomial_ci(data, "normal_appx", 90, None)['upr'] - binomial_ci(data, "normal_appx", 90, None)['lwr'], 3)}} | {{round(binomial_ci(data, "clopper_pearson", 90, None)['upr'] - binomial_ci(data, "clopper_pearson", 90, None)['lwr'], 3)}} | {{round(binomial_ci(data, "Jeffrey", 90, None)['upr'] - binomial_ci(data, "Jeffrey", 90, None)['lwr'], 3)}} | {{round(binomial_ci(data, "Agresti_Coull", 90, None)['upr'] - binomial_ci(data, "Agresti_Coull", 90, None)['lwr'], 3)}} |
| 95\% | {{round(estimate_ci(data, 95, None)['upr'] - estimate_ci(data, 95, None)['lwr'], 3)}} | {{round(binomial_ci(data, "normal_appx", 95, None)['upr'] - binomial_ci(data, "normal_appx", 95, None)['lwr'], 3)}} | {{round(binomial_ci(data, "clopper_pearson", 95, None)['upr'] - binomial_ci(data, "clopper_pearson", 95, None)['lwr'], 3)}} | {{round(binomial_ci(data, "Jeffrey", 95, None)['upr'] - binomial_ci(data, "Jeffrey", 95, None)['lwr'], 3)}} | {{round(binomial_ci(data, "Agresti_Coull", 95, None)['upr'] - binomial_ci(data, "Agresti_Coull", 95, None)['lwr'], 3)}} |
| 99\% | {{round(estimate_ci(data, 99, None)['upr'] - estimate_ci(data, 99, None)['lwr'], 3)}} | {{round(binomial_ci(data, "normal_appx", 99, None)['upr'] - binomial_ci(data, "normal_appx", 99, None)['lwr'], 3)}} | {{round(binomial_ci(data, "clopper_pearson", 99, None)['upr'] - binomial_ci(data, "clopper_pearson", 99, None)['lwr'], 3)}} | {{round(binomial_ci(data, "Jeffrey", 99, None)['upr'] - binomial_ci(data, "Jeffrey", 99, None)['lwr'], 3)}} | {{round(binomial_ci(data, "Agresti_Coull", 99, None)['upr'] - binomial_ci(data, "Agresti_Coull", 99, None)['lwr'], 3)}} |
<!-- #endregion -->

The Jeffrey’s intervals have smallest width.
