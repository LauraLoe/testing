+++ 
title = "Deep Learning for Survival analysis" 
date = '2020-02-01' 
tags = [ "Deep Learning", "Neural Networks", "Class19/20",]
categories = ["course projects"] 
author = "Seminar Information Systems WS19/20" 
disqusShortname = "https-humbodt-wi-github-io-blog" 
description = "Deep Learning in Survival Analysis"
+++


# Deep Learning for Survival Analysis
<br>
#### Authors: Laura Löschmann, Daria Smorodina

---

## Table of content
1. [Motivation](#motivation) <br />
2. [Basics of Survival Analysis](#introduction_sa) <br />
2.1 [Common terms](#terms) <br />
2.2 [Survival function](#survival_function) <br />
2.3 [Hazard function](#hazard_function) <br />
3. [Dataset](#dataset) <br />
4. [Standard Methods in Survival Analysis](#standard_methods) <br />
4.1 [Kaplan-Meier estimate](#kmf) <br />
4.2 [Cox Proportional Hazard Model](#coxph) <br />
4.3 [Time-Varying Cox Regression Model](#time_cox) <br />
4.4 [Random Survival Forests](#rsf) <br />
5. [Deep Learning in Survival Analysis](#deeplearning_sa) <br />
5.1 [DeepSurv](#deepsurv) <br />
5.2 [Deep Hit](#deephit) <br />
6. [Evaluation](#evaluation)  <br />
7. [Conclusion](#conclusion) <br />
8. [References](#references) <br />

---

## 1. Motivation <a class="anchor" id="first"></a>
Survival analysis also called as time-to event analysis refers to the set of statistical analyses that takes a series of observations and attempts to estimate the time it takes for an event of interest to occur.
The development of survival analysis dates back to the 17th century with the first life table ever produced by English statistician John Graunt in 1662.  The name ‚Survival Analysis‘ comes from the longstanding application of these methods since throughout centuries they were solely linked to investigating mortality rates. However, during the last decades the applications of the statistical methods of survival analysis have been extended beyond medical research to other fields like health insurance, marketing, sociology and engineering.

Survival analysis attempts to answer questions like: What is the proportion of a population which will survive past a certain time? Of those that survive, at what rate will they die or fail? How long does it take for a machine to fail or a customer to churn? Can multiple causes of death or failure be taken into account? How do particular circumstances or characteristics increase or decrease the probability of survival? 

---

# 2. Basics of Survival Analysis <a class="anchor" id="second"></a>

## 2.a Common terms <a class="anchor" id="second_a"></a>
Survival analysis is a collection of data analysis methods with the outcome variable of interest ‚time to event‘. In general ‚event‘ describes the event of interest, also called **death event**, ‚time‘ refers to the point of time of first observation, also called **birth event**, and ‚time to event‘ is the **duration** between the first observation and the time the event occurs.
The subjects whose data were collected for survival analysis usually do not have the same time of first observation. A subject can enter the study at any time. Using durations ensure a necessary relativeness.

During the time of the study not every subject will experience the event of interest. Consequently it is unknown if the subjects will experience the event of interest in the future. The computation of the duration, the time from the first observation to the event of interest, is impossible. This special type kind of missing data can emerge due to three reasons:

1. The subject is still part of the study but has not experienced the event of interest yet.
2. The subject got lost during the study period and it is unknown if the subject experienced the event in the time of study.
3. The subject withdraws from the study due to some reasons.

In survival analysis this missing data is called **censorship** which refers to the inability to observe the variable of interest for the entire population. However, the censoring of data must be taken into account, dropping unobserved data would underestimate customer lifetimes and bias the results. Hence the particular subjects are labelled ‚censored‘.
Since for the censored subjects the death event could not be observed, the type of censorship is called right censoring which is the most common one in survival analysis. As opposed to this there is left censoring in case the birth event could not be observed. 

![](image.png) 

In terms of different application fields an exact determination of the birth and death event is vital.
Following there are a few examples of birth and death events as well as possible censoring cases for various use cases in industry:

Application field | Birth event | Death event | Censoring example
------------------|-------------|-------------|------------------
Predictive maintenance in mechanical operations|Time the machine was started for a continuous use|Time when the machine breaks down|Machine breaks down due to external reasons, e.g. a fire in the factory building
Customer analytics|Customer starts subscription|Time when customer churns or unsubscribe|Customer dies during the study
Medical research in terms of developing a heart disease|Time the subject was first observed|Time the subject developed a heart disease|Subject dies from a reason that has nothing to do with a heart disease
Lifetimes of political leaders around the world|Start of the tenure|Retirement|Leader dies during the tenure

---

## 2.b Survival Function<a class="anchor" id="second_b"></a>
The survival function S(t) defines the probability that the event of interest has not occurred at time t, or equivalently, the probability that the duration will be at least t. The survival function of a population is defined as follows:

$$S(t) = Pr(T > t)$$

T is the random lifetime taken from the population under study and cannot be negative. With regard to the dataset it is the amount of time a customer is able to pay his mortgage rates, he is not defaulting. The survival function S (t) outputs values between 0 and 1 and is a non-increasing function of t.
At the start of the study (t=0), no subject has experienced the event yet. Therefore the the probability S(0) of surviving beyond time 0 is one. In theory the survival function is smooth, in practice the events are observed on a concrete time scale, e.g. days, weeks, months, etc., such that the graph of the survival function is like a step function.

---

## 2.c Hazard Function<a class="anchor" id="second_c"></a>
The hazard function h(t) gives the probability of the death event occurring at time t, given that the subject did not experience the death event until time t. It describes the instantaneous potential per unit time for the event to occur.

$$h(t) = \lim_{\delta t\to 0}\frac{Pr(t≤T≤t+\delta t | T>t)}{\delta t}$$

In contrast to the survival function, the hazard function does not have to start at 1 and go down to 0. The hazard rate usually changes over time. It can start anywhere and go up and down over time. For instance the probability of defaulting on a mortgage may be low in the beginning but can increase over the time of the mortgage.

The main goal of survival analysis is to estimate and interpret survival and/or hazard functions from survival data. 

---

# 3. Intro into our dataset<a class="anchor" id="three"></a>

The dataset for the following survival analysis contains data of subjects that took on a mortgage. In this case the birth event is the time when the subject was first observed for the study and the death event is the default of the subject. Thw duration is the time between the birth and death event. The dataset does not contain any lost or withdrawn subjects but there exist subjects who have not defaulted yet. These subjects will be labelled ‚censored‘ in further analysis.


```python
import pandas as pd
from pandas import DataFrame

import numpy as np
import sklearn

import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.dates as mdates
plt.rcParams.update({'figure.figsize':(16,7), 'figure.dpi':100})
plt.style.use('seaborn-white')

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
```

### Load and explore the given dataset


```python
data = pd.read_csv('mortgage.csv', sep = ",")
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>time</th>
      <th>orig_time</th>
      <th>first_time</th>
      <th>mat_time</th>
      <th>balance_time</th>
      <th>LTV_time</th>
      <th>interest_rate_time</th>
      <th>hpi_time</th>
      <th>gdp_time</th>
      <th>...</th>
      <th>REtype_SF_orig_time</th>
      <th>investor_orig_time</th>
      <th>balance_orig_time</th>
      <th>FICO_orig_time</th>
      <th>LTV_orig_time</th>
      <th>Interest_Rate_orig_time</th>
      <th>hpi_orig_time</th>
      <th>default_time</th>
      <th>payoff_time</th>
      <th>status_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>25</td>
      <td>-7</td>
      <td>25</td>
      <td>113</td>
      <td>41303.42</td>
      <td>24.498336</td>
      <td>9.2</td>
      <td>226.29</td>
      <td>2.899137</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>45000.0</td>
      <td>715</td>
      <td>69.4</td>
      <td>9.2</td>
      <td>87.03</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>26</td>
      <td>-7</td>
      <td>25</td>
      <td>113</td>
      <td>41061.95</td>
      <td>24.483867</td>
      <td>9.2</td>
      <td>225.10</td>
      <td>2.151365</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>45000.0</td>
      <td>715</td>
      <td>69.4</td>
      <td>9.2</td>
      <td>87.03</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>27</td>
      <td>-7</td>
      <td>25</td>
      <td>113</td>
      <td>40804.42</td>
      <td>24.626795</td>
      <td>9.2</td>
      <td>222.39</td>
      <td>2.361722</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>45000.0</td>
      <td>715</td>
      <td>69.4</td>
      <td>9.2</td>
      <td>87.03</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>28</td>
      <td>-7</td>
      <td>25</td>
      <td>113</td>
      <td>40483.89</td>
      <td>24.735883</td>
      <td>9.2</td>
      <td>219.67</td>
      <td>1.229172</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>45000.0</td>
      <td>715</td>
      <td>69.4</td>
      <td>9.2</td>
      <td>87.03</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>29</td>
      <td>-7</td>
      <td>25</td>
      <td>113</td>
      <td>40367.06</td>
      <td>24.925476</td>
      <td>9.2</td>
      <td>217.37</td>
      <td>1.692969</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>45000.0</td>
      <td>715</td>
      <td>69.4</td>
      <td>9.2</td>
      <td>87.03</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
print("Maximal lifetime (in days) in dataset: ", data['time'].max())
```

    Maximal lifetime (in days) in dataset:  60
    


```python
data.shape
```




    (622489, 23)



### Counting the risks


```python
data['default_time'].value_counts()
```




    0    607331
    1     15158
    Name: default_time, dtype: int64




```python
data['payoff_time'].value_counts()
```




    0    595900
    1     26589
    Name: payoff_time, dtype: int64




```python
data['status_time'].value_counts()
```




    0    580742
    2     26589
    1     15158
    Name: status_time, dtype: int64



### Checking the missing values in dataset and replacing them with means


```python
# Check for missing values
data.isnull().sum()
```




    id                           0
    time                         0
    orig_time                    0
    first_time                   0
    mat_time                     0
    balance_time                 0
    LTV_time                   270
    interest_rate_time           0
    hpi_time                     0
    gdp_time                     0
    uer_time                     0
    REtype_CO_orig_time          0
    REtype_PU_orig_time          0
    REtype_SF_orig_time          0
    investor_orig_time           0
    balance_orig_time            0
    FICO_orig_time               0
    LTV_orig_time                0
    Interest_Rate_orig_time      0
    hpi_orig_time                0
    default_time                 0
    payoff_time                  0
    status_time                  0
    dtype: int64



### Rename columns for more convenient usage


```python
data = data.rename(columns={"orig_time": "origination_time", "mat_time": "maturity_time",
                            "hpi_time" : "house_price_index_time",  "hpi_orig_time" : "house_price_index_orig_time", 
                            "uer_time" : "unemployment_rate_time", 
                            "REtype_CO_orig_time" : "real_estate_condominium", 
                            "REtype_PU_orig_time" : "real_estate_planned_urban_dev", 
                            "REtype_SF_orig_time" : "real_estate_single_family_home", "Interest_Rate_orig_time" : "interest_rate_orig_time"})
```

---

- ## Data preprocessing

### Max time for each borrower


```python
time_max = data.groupby(['id']).agg({'time' : 'max'}).reset_index()
time_max.rename(columns = {'time' : 'time_max'}, inplace = True)
```


```python
data = pd.merge(data,time_max, on ='id')
```


```python
time_max[time_max.columns[1:]].mean()
```




    time_max    36.17074
    dtype: float64




```python
time_max.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>time_max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
</div>



### Duration column


```python
data['total_obs_time'] = data.apply(lambda r: int(r['time_max'] - r['first_time']+1), axis = 1)
```

### Missing values imputation


```python
data['LTV_time'].fillna((data['LTV_time'].mean()),inplace=True)
```

### Backup Copy


```python
data_all = data.copy()
```


```python
df = data.copy()
```


```python
df = df[df.id != df.id.shift(-1)]
```

### Default event distribution graph


```python
df_map = df.copy()

mapping = {0: 'Censored', 1: 'Event Occured'}
df_map = df_map.replace({'default_time': mapping})
```


```python
def_risk  = df_map['default_time'].value_counts()
```


```python
cens = df_map['default_time'].value_counts()[0]
ev = df_map['default_time'].value_counts()[1]
cens_per = cens / df_map.shape[0] * 100
ev_per = ev / df_map.shape[0] * 100

plt.figure(figsize=(10, 8))
sns.countplot(df_map['default_time'])

plt.xlabel('', size=15, labelpad=15)
plt.ylabel('Frequency', size=15, labelpad=15)
plt.xticks((1, 0), ['Censored ({0:.2f}%)'.format(cens_per), 'Event Occured ({0:.2f}%)'.format(ev_per)])
plt.tick_params(axis='x', labelsize=13)
plt.tick_params(axis='y', labelsize=13)

plt.title('Event Distribution', size=15, y=1.05)

plt.show()
```


![png](output_49_0.png)



```python
df['default_time'].value_counts()
```




    0    34846
    1    15154
    Name: default_time, dtype: int64




```python
df['payoff_time'].value_counts()
```




    1    26589
    0    23411
    Name: payoff_time, dtype: int64




```python
df['status_time'].value_counts()
```




    2    26589
    1    15154
    0     8257
    Name: status_time, dtype: int64




```python
'''
cat_features = ['investor_orig_time', 'real_estate_single_family_home', 'real_estate_planned_urban_dev']

fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(20, 20))
plt.subplots_adjust(right=1.5, top=1.25)

for i, feature in enumerate(cat_features, 1):    
    plt.subplot(2, 3, i)
    sns.countplot(x=feature, hue='default_time', data=df)
    
    plt.xlabel('{}'.format(feature), size=20, labelpad=15)
    plt.ylabel('Borrower Count', size=20, labelpad=15)    
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    
    plt.legend(['Not Defaulted', 'Defaulted'], loc='upper center', prop={'size': 18})
    plt.title('Count of Defaults in {} Feature'.format(feature), size=20, y=1.05)

plt.show()
'''
```




    "\ncat_features = ['investor_orig_time', 'real_estate_single_family_home', 'real_estate_planned_urban_dev']\n\nfig, axs = plt.subplots(ncols=3, nrows=1, figsize=(20, 20))\nplt.subplots_adjust(right=1.5, top=1.25)\n\nfor i, feature in enumerate(cat_features, 1):    \n    plt.subplot(2, 3, i)\n    sns.countplot(x=feature, hue='default_time', data=df)\n    \n    plt.xlabel('{}'.format(feature), size=20, labelpad=15)\n    plt.ylabel('Borrower Count', size=20, labelpad=15)    \n    plt.tick_params(axis='x', labelsize=20)\n    plt.tick_params(axis='y', labelsize=20)\n    \n    plt.legend(['Not Defaulted', 'Defaulted'], loc='upper center', prop={'size': 18})\n    plt.title('Count of Defaults in {} Feature'.format(feature), size=20, y=1.05)\n\nplt.show()\n"



### Dropping out some columns


```python
data_all = data_all.drop(['status_time', 'first_time', 'time_max', 'payoff_time'], axis = 1)

data_one = df.copy()
data_one = data_one.drop(['time', 'status_time', 'first_time', 'time_max', 'payoff_time'], axis = 1)
```


```python
print(data_all.shape, data_one.shape)
```

    (622489, 21) (50000, 20)
    


```python
data_cox = data_one.copy()
```

### Additional: Censorhip plot


```python
current_time = 12
```


```python
actual_lifetimes = data_one.total_obs_time[:30]
actual_lifetimes.index = np.arange(1, len(actual_lifetimes)+1)
```


```python
actual_lifetimes.head()
```




    1    24
    2     2
    3     5
    4    36
    5     3
    Name: total_obs_time, dtype: int64




```python
observed_lifetimes = np.minimum(actual_lifetimes, current_time)
observed_lifetimes.head()
```




    1    12
    2     2
    3     5
    4    12
    5     3
    Name: total_obs_time, dtype: int64




```python
death_observed = actual_lifetimes < current_time
death_observed.head()
```




    1    False
    2     True
    3     True
    4    False
    5     True
    Name: total_obs_time, dtype: bool




```python
from lifelines.plotting import plot_lifetimes

ax = plot_lifetimes(actual_lifetimes, event_observed=death_observed)
ax.set_xlim(0, 35)
ax.vlines(10, 0, len(actual_lifetimes), lw=2, linestyles='--')
ax.set_ylabel('Borrowers', fontsize = 15)
ax.set_xlabel("Time (Months)", fontsize = 15)
ax.set_title("Default events for our individuals, at $t=10$", fontsize = 15)
```

    C:\Users\frusi\Anaconda3\envs\sis\lib\site-packages\pandas\core\series.py:1146: FutureWarning: 
    Passing list-likes to .loc or [] with any missing label will raise
    KeyError in the future, you can use .reindex() as an alternative.
    
    See the documentation here:
    https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike
      return self.loc[key]
    




    Text(0.5, 1.0, 'Default events for our individuals, at $t=10$')




![png](output_64_2.png)


---

# 4. Standard Methods in Survival Analysis<a class="anchor" id="four"></a>


```python
scaler = StandardScaler()
```


```python
data_cox = data_cox.set_index('id')
```


```python
'''
xx = ['balance_time', 'LTV_time',
       'interest_rate_time', 'house_price_index_time', 'gdp_time', 'unemployment_rate_time',
       'balance_orig_time', 'FICO_orig_time',
       'LTV_orig_time', 'interest_rate_orig_time', 'house_price_index_orig_time']

data_cox[xx] = scaler.fit_transform(data_cox[xx])
'''
```




    "\nxx = ['balance_time', 'LTV_time',\n       'interest_rate_time', 'house_price_index_time', 'gdp_time', 'unemployment_rate_time',\n       'balance_orig_time', 'FICO_orig_time',\n       'LTV_orig_time', 'interest_rate_orig_time', 'house_price_index_orig_time']\n\ndata_cox[xx] = scaler.fit_transform(data_cox[xx])\n"



## 4.a Kaplan Meier Estimator<a class="anchor" id="four_a"></a>


```python
import lifelines
```

### Fitting the Kaplan-Meier estimate for the survival function


```python
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()
```


```python
T = data_cox["total_obs_time"]
E = data_cox["default_time"]

kmf.fit(T, event_observed=E)
```




    <lifelines.KaplanMeierFitter:"KM_estimate", fitted with 50000 total observations, 34846 right-censored observations>



### Plot for KMF


```python
kmf.plot()

plt.title("The Kaplan-Meier Estimate", fontsize = 15)
plt.ylabel("Probability a Borrower is not defaulted", fontsize = 15)

plt.show()
```


![png](output_76_0.png)


### The estimated median time to event

Return the unique time point, t, such that S(t) = 0.5. This is the “half-life” of the population, and a robust summary statistic for the population, if it exists.


```python
kmf.median_survival_time_
```




    26.0



### A summary of the life table


```python
kmf.event_table[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>removed</th>
      <th>observed</th>
      <th>censored</th>
      <th>entrance</th>
      <th>at_risk</th>
    </tr>
    <tr>
      <th>event_at</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>50000</td>
      <td>50000</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>3773</td>
      <td>623</td>
      <td>3150</td>
      <td>0</td>
      <td>50000</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>3992</td>
      <td>892</td>
      <td>3100</td>
      <td>0</td>
      <td>46227</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>3693</td>
      <td>999</td>
      <td>2694</td>
      <td>0</td>
      <td>42235</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>3557</td>
      <td>965</td>
      <td>2592</td>
      <td>0</td>
      <td>38542</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>2971</td>
      <td>996</td>
      <td>1975</td>
      <td>0</td>
      <td>34985</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>2768</td>
      <td>953</td>
      <td>1815</td>
      <td>0</td>
      <td>32014</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>3029</td>
      <td>936</td>
      <td>2093</td>
      <td>0</td>
      <td>29246</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>2583</td>
      <td>985</td>
      <td>1598</td>
      <td>0</td>
      <td>26217</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>3177</td>
      <td>894</td>
      <td>2283</td>
      <td>0</td>
      <td>23634</td>
    </tr>
    <tr>
      <th>10.0</th>
      <td>1444</td>
      <td>766</td>
      <td>678</td>
      <td>0</td>
      <td>20457</td>
    </tr>
  </tbody>
</table>
</div>



### Estimated survival function


```python
kmf_surv = kmf.survival_function_
```


```python
kmf_surv[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>KM_estimate</th>
    </tr>
    <tr>
      <th>timeline</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.987540</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>0.968484</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>0.945576</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>0.921901</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>0.895656</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>0.868993</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>0.841182</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>0.809578</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>0.778954</td>
    </tr>
    <tr>
      <th>10.0</th>
      <td>0.749787</td>
    </tr>
  </tbody>
</table>
</div>



---

## 4.b Cox Proportional Hazard Model<a class="anchor" id="four_b"></a>


```python
from lifelines import CoxPHFitter

cph = CoxPHFitter(penalizer=0.1)
cph.fit(data_cox, duration_col='total_obs_time', event_col='default_time', show_progress = True)
```

    Iteration 1: norm_delta = 1.07434, step_size = 0.9500, ll = -151616.37256, newton_decrement = 7849.88388, seconds_since_start = 0.0
    Iteration 2: norm_delta = 2.66941, step_size = 0.9500, ll = -157315.04540, newton_decrement = 11492.51839, seconds_since_start = 0.1
    Iteration 3: norm_delta = 0.44479, step_size = 0.9500, ll = -146405.50893, newton_decrement = 1879.77130, seconds_since_start = 0.1
    Iteration 4: norm_delta = 0.07525, step_size = 0.9310, ll = -144506.16389, newton_decrement = 235.40711, seconds_since_start = 0.1
    Iteration 5: norm_delta = 0.05009, step_size = 1.0000, ll = -144222.24240, newton_decrement = 49.75079, seconds_since_start = 0.1
    Iteration 6: norm_delta = 0.02288, step_size = 1.0000, ll = -144163.67104, newton_decrement = 5.93358, seconds_since_start = 0.2
    Iteration 7: norm_delta = 0.00403, step_size = 1.0000, ll = -144157.19919, newton_decrement = 0.13884, seconds_since_start = 0.2
    Iteration 8: norm_delta = 0.00011, step_size = 1.0000, ll = -144157.05760, newton_decrement = 0.00009, seconds_since_start = 0.2
    Iteration 9: norm_delta = 0.00000, step_size = 1.0000, ll = -144157.05750, newton_decrement = 0.00000, seconds_since_start = 0.3
    Convergence success after 9 iterations.
    




    <lifelines.CoxPHFitter: fitted with 50000 total observations, 34846 right-censored observations>



### Filter down to just censored subjects to predict remaining survival


```python
censored_subjects = data_cox.loc[~data_cox['default_time'].astype(bool)]
censored_subjects_last_obs = censored_subjects['total_obs_time']
```


```python
censored_subjects.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origination_time</th>
      <th>maturity_time</th>
      <th>balance_time</th>
      <th>LTV_time</th>
      <th>interest_rate_time</th>
      <th>house_price_index_time</th>
      <th>gdp_time</th>
      <th>unemployment_rate_time</th>
      <th>real_estate_condominium</th>
      <th>real_estate_planned_urban_dev</th>
      <th>real_estate_single_family_home</th>
      <th>investor_orig_time</th>
      <th>balance_orig_time</th>
      <th>FICO_orig_time</th>
      <th>LTV_orig_time</th>
      <th>interest_rate_orig_time</th>
      <th>house_price_index_orig_time</th>
      <th>default_time</th>
      <th>total_obs_time</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>18</td>
      <td>138</td>
      <td>105654.77</td>
      <td>65.469851</td>
      <td>7.680</td>
      <td>225.10</td>
      <td>2.151365</td>
      <td>4.7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>107200.0</td>
      <td>558</td>
      <td>80.0</td>
      <td>7.680</td>
      <td>186.91</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-6</td>
      <td>114</td>
      <td>44378.60</td>
      <td>31.459735</td>
      <td>11.375</td>
      <td>217.37</td>
      <td>1.692969</td>
      <td>4.5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>48600.0</td>
      <td>680</td>
      <td>83.6</td>
      <td>8.750</td>
      <td>89.58</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2</td>
      <td>119</td>
      <td>52686.35</td>
      <td>34.898842</td>
      <td>10.500</td>
      <td>189.82</td>
      <td>2.836358</td>
      <td>5.7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>63750.0</td>
      <td>587</td>
      <td>81.8</td>
      <td>10.500</td>
      <td>97.99</td>
      <td>0</td>
      <td>36</td>
    </tr>
    <tr>
      <th>5</th>
      <td>18</td>
      <td>138</td>
      <td>52100.71</td>
      <td>66.346343</td>
      <td>9.155</td>
      <td>222.39</td>
      <td>2.361722</td>
      <td>4.4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>52800.0</td>
      <td>527</td>
      <td>80.0</td>
      <td>9.155</td>
      <td>186.91</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>18</td>
      <td>138</td>
      <td>107916.38</td>
      <td>77.919574</td>
      <td>9.000</td>
      <td>225.10</td>
      <td>2.151365</td>
      <td>4.7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>109250.0</td>
      <td>601</td>
      <td>95.0</td>
      <td>9.000</td>
      <td>186.91</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
cph.predict_partial_hazard(censored_subjects).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1.745313</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.591282</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.412496</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.447832</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2.246407</td>
    </tr>
  </tbody>
</table>
</div>




```python
cph.predict_survival_function(censored_subjects, times=[5., 25., 50.], conditional_after=censored_subjects_last_obs)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>7</th>
      <th>8</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>...</th>
      <th>49991</th>
      <th>49992</th>
      <th>49993</th>
      <th>49994</th>
      <th>49995</th>
      <th>49996</th>
      <th>49997</th>
      <th>49998</th>
      <th>49999</th>
      <th>50000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5.0</th>
      <td>0.861492</td>
      <td>0.930291</td>
      <td>0.970440</td>
      <td>0.789772</td>
      <td>0.825394</td>
      <td>0.944683</td>
      <td>0.791579</td>
      <td>0.905416</td>
      <td>0.885518</td>
      <td>0.783490</td>
      <td>...</td>
      <td>0.952365</td>
      <td>0.945352</td>
      <td>0.980546</td>
      <td>0.965394</td>
      <td>0.964302</td>
      <td>0.961157</td>
      <td>0.952614</td>
      <td>0.967566</td>
      <td>0.978198</td>
      <td>0.982348</td>
    </tr>
    <tr>
      <th>25.0</th>
      <td>0.276602</td>
      <td>0.645134</td>
      <td>0.938723</td>
      <td>0.163953</td>
      <td>0.191252</td>
      <td>0.582798</td>
      <td>0.166848</td>
      <td>0.467085</td>
      <td>0.393964</td>
      <td>0.154222</td>
      <td>...</td>
      <td>0.821048</td>
      <td>0.796894</td>
      <td>0.923701</td>
      <td>0.867377</td>
      <td>0.863421</td>
      <td>0.852102</td>
      <td>0.821917</td>
      <td>0.875289</td>
      <td>0.914799</td>
      <td>0.930579</td>
    </tr>
    <tr>
      <th>50.0</th>
      <td>0.167542</td>
      <td>0.561215</td>
      <td>0.938723</td>
      <td>0.084516</td>
      <td>0.100315</td>
      <td>0.463636</td>
      <td>0.086562</td>
      <td>0.353378</td>
      <td>0.280030</td>
      <td>0.077737</td>
      <td>...</td>
      <td>0.780298</td>
      <td>0.751528</td>
      <td>0.904966</td>
      <td>0.836094</td>
      <td>0.831299</td>
      <td>0.817610</td>
      <td>0.781338</td>
      <td>0.845701</td>
      <td>0.894008</td>
      <td>0.913453</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 34846 columns</p>
</div>



### Plotting the coefficients


```python
cph.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249b47dab38>




![png](output_94_1.png)


### Plotting the effect of varying a covariate

After fitting, we can plot what the survival curves look like as we vary a single covariate while holding everything else equal. This is useful to understand the impact of a covariate, given the model. To do this, we use the plot_covariate_groups() method and give it the covariate of interest, and the values to display.


```python
cph.plot_covariate_groups('real_estate_single_family_home', values=range(0, 15, 3), cmap='coolwarm',
                         plot_baseline=True)

plt.title("The CoxPH Model for Variable: Real Estate Single Family Home", fontsize = 15)
plt.ylabel("Probability a Borrower is not defaulted", fontsize = 15)

plt.show()
```


![png](output_97_0.png)



```python
data_cox.columns
```




    Index(['origination_time', 'maturity_time', 'balance_time', 'LTV_time',
           'interest_rate_time', 'house_price_index_time', 'gdp_time',
           'unemployment_rate_time', 'real_estate_condominium',
           'real_estate_planned_urban_dev', 'real_estate_single_family_home',
           'investor_orig_time', 'balance_orig_time', 'FICO_orig_time',
           'LTV_orig_time', 'interest_rate_orig_time',
           'house_price_index_orig_time', 'default_time', 'total_obs_time'],
          dtype='object')




```python
cph.plot_covariate_groups(
    ['real_estate_planned_urban_dev', 'real_estate_single_family_home'],
    np.eye(2),
    cmap='coolwarm')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249b441def0>




![png](output_99_1.png)



```python
cph.baseline_hazard_[:15]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>baseline hazard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.0</th>
      <td>0.007255</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>0.011592</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>0.014221</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>0.015132</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>0.017309</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>0.018385</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>0.020376</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>0.025215</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>0.027927</td>
    </tr>
    <tr>
      <th>10.0</th>
      <td>0.030304</td>
    </tr>
    <tr>
      <th>11.0</th>
      <td>0.032976</td>
    </tr>
    <tr>
      <th>12.0</th>
      <td>0.034718</td>
    </tr>
    <tr>
      <th>13.0</th>
      <td>0.036830</td>
    </tr>
    <tr>
      <th>14.0</th>
      <td>0.039296</td>
    </tr>
    <tr>
      <th>15.0</th>
      <td>0.037979</td>
    </tr>
  </tbody>
</table>
</div>



### Concordance index

concordance-index, also known as the c-index. This measure evaluates the accuracy of the ranking of predicted time. It is in fact a generalization of AUC, another common loss function, and is interpreted similarly:

0.5 is the expected result from random predictions,
1.0 is perfect concordance and,
0.0 is perfect anti-concordance (multiply predictions with -1 to get 1.0)


```python
cph.score_
```




    0.7764777518247834



or


```python
from lifelines.utils import concordance_index

cindex_cox = concordance_index(data_cox['total_obs_time'], -cph.predict_partial_hazard(data_cox), data_cox['default_time'])
```


```python
cindex_cox
```




    0.7764777518247834



### The estimated coefficients

Changed in version 0.22.0: use to be .hazards_


```python
cph.params_
```




    origination_time                  0.029912
    maturity_time                     0.008111
    balance_time                      0.000007
    LTV_time                          0.004866
    interest_rate_time                0.213809
    house_price_index_time            0.008434
    gdp_time                         -0.248223
    unemployment_rate_time           -0.056415
    real_estate_condominium           0.114276
    real_estate_planned_urban_dev     0.091901
    real_estate_single_family_home    0.034783
    investor_orig_time                0.061815
    balance_orig_time                -0.000007
    FICO_orig_time                   -0.002951
    LTV_orig_time                     0.003541
    interest_rate_orig_time          -0.013820
    house_price_index_orig_time       0.004847
    dtype: float64



attribute is available which is the exponentiation of params_


```python
cph.hazard_ratios_
```




    origination_time                  1.030364
    maturity_time                     1.008144
    balance_time                      1.000007
    LTV_time                          1.004878
    interest_rate_time                1.238386
    house_price_index_time            1.008470
    gdp_time                          0.780186
    unemployment_rate_time            0.945147
    real_estate_condominium           1.121061
    real_estate_planned_urban_dev     1.096256
    real_estate_single_family_home    1.035395
    investor_orig_time                1.063765
    balance_orig_time                 0.999993
    FICO_orig_time                    0.997054
    LTV_orig_time                     1.003547
    interest_rate_orig_time           0.986275
    house_price_index_orig_time       1.004858
    Name: exp(coef), dtype: float64




```python
cph.check_assumptions(data_cox)
```

    The ``p_value_threshold`` is set at 0.01. Even under the null hypothesis of no violations, some
    covariates will be below the threshold by chance. This is compounded when there are many covariates.
    Similarly, when there are lots of observations, even minor deviances from the proportional hazard
    assumption will be flagged.
    
    With that in mind, it's best to use a combination of statistical tests and visual tests to determine
    the most serious violations. Produce visual plots using ``check_assumptions(..., show_plots=True)``
    and looking for non-constant lines. See link [A] below for a full example.
    
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>null_distribution</th>
      <td>chi squared</td>
    </tr>
    <tr>
      <th>degrees_of_freedom</th>
      <td>1</td>
    </tr>
    <tr>
      <th>test_name</th>
      <td>proportional_hazard_test</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>test_statistic</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">FICO_orig_time</th>
      <th>km</th>
      <td>36.97</td>
      <td>&lt;0.005</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>53.22</td>
      <td>&lt;0.005</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">LTV_orig_time</th>
      <th>km</th>
      <td>11.15</td>
      <td>&lt;0.005</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>12.45</td>
      <td>&lt;0.005</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">LTV_time</th>
      <th>km</th>
      <td>0.20</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>0.29</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">balance_orig_time</th>
      <th>km</th>
      <td>3.28</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>3.67</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">balance_time</th>
      <th>km</th>
      <td>2.14</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>2.45</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">gdp_time</th>
      <th>km</th>
      <td>230.17</td>
      <td>&lt;0.005</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>109.89</td>
      <td>&lt;0.005</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">house_price_index_orig_time</th>
      <th>km</th>
      <td>0.00</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>0.40</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">house_price_index_time</th>
      <th>km</th>
      <td>291.24</td>
      <td>&lt;0.005</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>265.76</td>
      <td>&lt;0.005</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">interest_rate_orig_time</th>
      <th>km</th>
      <td>524.86</td>
      <td>&lt;0.005</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>574.26</td>
      <td>&lt;0.005</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">interest_rate_time</th>
      <th>km</th>
      <td>12.17</td>
      <td>&lt;0.005</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>26.13</td>
      <td>&lt;0.005</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">investor_orig_time</th>
      <th>km</th>
      <td>21.62</td>
      <td>&lt;0.005</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>24.01</td>
      <td>&lt;0.005</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">maturity_time</th>
      <th>km</th>
      <td>0.02</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>0.00</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">origination_time</th>
      <th>km</th>
      <td>143.17</td>
      <td>&lt;0.005</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>104.38</td>
      <td>&lt;0.005</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">real_estate_condominium</th>
      <th>km</th>
      <td>0.71</td>
      <td>0.40</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>0.26</td>
      <td>0.61</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">real_estate_planned_urban_dev</th>
      <th>km</th>
      <td>0.93</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>2.40</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">real_estate_single_family_home</th>
      <th>km</th>
      <td>3.90</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>5.16</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">unemployment_rate_time</th>
      <th>km</th>
      <td>2112.17</td>
      <td>&lt;0.005</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>1918.03</td>
      <td>&lt;0.005</td>
    </tr>
  </tbody>
</table>


    
    
    1. Variable 'origination_time' failed the non-proportional test: p-value is <5e-05.
    
       Advice 1: the functional form of the variable 'origination_time' might be incorrect. That is,
    there may be non-linear terms missing. The proportional hazard test used is very sensitive to
    incorrect functional forms. See documentation in link [D] below on how to specify a functional form.
    
       Advice 2: try binning the variable 'origination_time' using pd.cut, and then specify it in
    `strata=['origination_time', ...]` in the call in `.fit`. See documentation in link [B] below.
    
       Advice 3: try adding an interaction term with your time variable. See documentation in link [C]
    below.
    
    
    2. Variable 'interest_rate_time' failed the non-proportional test: p-value is <5e-05.
    
       Advice 1: the functional form of the variable 'interest_rate_time' might be incorrect. That is,
    there may be non-linear terms missing. The proportional hazard test used is very sensitive to
    incorrect functional forms. See documentation in link [D] below on how to specify a functional form.
    
       Advice 2: try binning the variable 'interest_rate_time' using pd.cut, and then specify it in
    `strata=['interest_rate_time', ...]` in the call in `.fit`. See documentation in link [B] below.
    
       Advice 3: try adding an interaction term with your time variable. See documentation in link [C]
    below.
    
    
    3. Variable 'house_price_index_time' failed the non-proportional test: p-value is <5e-05.
    
       Advice 1: the functional form of the variable 'house_price_index_time' might be incorrect. That
    is, there may be non-linear terms missing. The proportional hazard test used is very sensitive to
    incorrect functional forms. See documentation in link [D] below on how to specify a functional form.
    
       Advice 2: try binning the variable 'house_price_index_time' using pd.cut, and then specify it in
    `strata=['house_price_index_time', ...]` in the call in `.fit`. See documentation in link [B] below.
    
       Advice 3: try adding an interaction term with your time variable. See documentation in link [C]
    below.
    
    
    4. Variable 'gdp_time' failed the non-proportional test: p-value is <5e-05.
    
       Advice 1: the functional form of the variable 'gdp_time' might be incorrect. That is, there may
    be non-linear terms missing. The proportional hazard test used is very sensitive to incorrect
    functional forms. See documentation in link [D] below on how to specify a functional form.
    
       Advice 2: try binning the variable 'gdp_time' using pd.cut, and then specify it in
    `strata=['gdp_time', ...]` in the call in `.fit`. See documentation in link [B] below.
    
       Advice 3: try adding an interaction term with your time variable. See documentation in link [C]
    below.
    
    
    5. Variable 'unemployment_rate_time' failed the non-proportional test: p-value is <5e-05.
    
       Advice 1: the functional form of the variable 'unemployment_rate_time' might be incorrect. That
    is, there may be non-linear terms missing. The proportional hazard test used is very sensitive to
    incorrect functional forms. See documentation in link [D] below on how to specify a functional form.
    
       Advice 2: try binning the variable 'unemployment_rate_time' using pd.cut, and then specify it in
    `strata=['unemployment_rate_time', ...]` in the call in `.fit`. See documentation in link [B] below.
    
       Advice 3: try adding an interaction term with your time variable. See documentation in link [C]
    below.
    
    
    6. Variable 'investor_orig_time' failed the non-proportional test: p-value is <5e-05.
    
       Advice: with so few unique values (only 2), you can include `strata=['investor_orig_time', ...]`
    in the call in `.fit`. See documentation in link [E] below.
    
    7. Variable 'FICO_orig_time' failed the non-proportional test: p-value is <5e-05.
    
       Advice 1: the functional form of the variable 'FICO_orig_time' might be incorrect. That is, there
    may be non-linear terms missing. The proportional hazard test used is very sensitive to incorrect
    functional forms. See documentation in link [D] below on how to specify a functional form.
    
       Advice 2: try binning the variable 'FICO_orig_time' using pd.cut, and then specify it in
    `strata=['FICO_orig_time', ...]` in the call in `.fit`. See documentation in link [B] below.
    
       Advice 3: try adding an interaction term with your time variable. See documentation in link [C]
    below.
    
    
    8. Variable 'LTV_orig_time' failed the non-proportional test: p-value is 0.0004.
    
       Advice 1: the functional form of the variable 'LTV_orig_time' might be incorrect. That is, there
    may be non-linear terms missing. The proportional hazard test used is very sensitive to incorrect
    functional forms. See documentation in link [D] below on how to specify a functional form.
    
       Advice 2: try binning the variable 'LTV_orig_time' using pd.cut, and then specify it in
    `strata=['LTV_orig_time', ...]` in the call in `.fit`. See documentation in link [B] below.
    
       Advice 3: try adding an interaction term with your time variable. See documentation in link [C]
    below.
    
    
    9. Variable 'interest_rate_orig_time' failed the non-proportional test: p-value is <5e-05.
    
       Advice 1: the functional form of the variable 'interest_rate_orig_time' might be incorrect. That
    is, there may be non-linear terms missing. The proportional hazard test used is very sensitive to
    incorrect functional forms. See documentation in link [D] below on how to specify a functional form.
    
       Advice 2: try binning the variable 'interest_rate_orig_time' using pd.cut, and then specify it in
    `strata=['interest_rate_orig_time', ...]` in the call in `.fit`. See documentation in link [B]
    below.
    
       Advice 3: try adding an interaction term with your time variable. See documentation in link [C]
    below.
    
    
    ---
    [A]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html
    [B]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Bin-variable-and-stratify-on-it
    [C]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Introduce-time-varying-covariates
    [D]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Modify-the-functional-form
    [E]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Stratification
    
    


```python
#cox_strata_duration = data_cox.copy()
#cox_strata_duration['dur_strata'] = pd.cut(cox_strata_duration['total_obs_time'], np.arange(0, 80, 3))
#cox_strata_duration[['total_obs_time', 'dur_strata']].head(10)
```

### Cross - validation


```python
from lifelines.utils import k_fold_cross_validation
```


```python
scores = k_fold_cross_validation(cph, data_cox, 'total_obs_time', event_col='default_time', k=6)
print("Print all scores in CV: ", scores)
print(' ')
print("Mean c-index: ", np.mean(scores))
print("Std: ", np.std(scores))
```

    Print all scores in CV:  [0.7932822945633529, 0.77646299524676, 0.780871983194206, 0.7736959995988707, 0.7733554968290542, 0.7755513037778162]
     
    Mean c-index:  0.7788700122016766
    Std:  0.006900599166743354
    

#### Some useful functions from lifelines.utils

https://lifelines.readthedocs.io/en/latest/lifelines.utils.html

---

## 4.c Time-Varying Cox Regression Model<a class="anchor" id="four_c"></a>


```python
data_cox_time = data_all.copy()
```


```python
data_cox_time = data_cox_time.drop('total_obs_time', axis = 1)
```

### Data preprocessing into long format


```python
from lifelines.utils import to_long_format

data_cox_tv = to_long_format(data_cox_time, duration_col = "time")
```


```python
data_cox_tv.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>origination_time</th>
      <th>maturity_time</th>
      <th>balance_time</th>
      <th>LTV_time</th>
      <th>interest_rate_time</th>
      <th>house_price_index_time</th>
      <th>gdp_time</th>
      <th>unemployment_rate_time</th>
      <th>real_estate_condominium</th>
      <th>...</th>
      <th>real_estate_single_family_home</th>
      <th>investor_orig_time</th>
      <th>balance_orig_time</th>
      <th>FICO_orig_time</th>
      <th>LTV_orig_time</th>
      <th>interest_rate_orig_time</th>
      <th>house_price_index_orig_time</th>
      <th>default_time</th>
      <th>start</th>
      <th>stop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>-7</td>
      <td>113</td>
      <td>41303.42</td>
      <td>24.498336</td>
      <td>9.2</td>
      <td>226.29</td>
      <td>2.899137</td>
      <td>4.7</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>45000.0</td>
      <td>715</td>
      <td>69.4</td>
      <td>9.2</td>
      <td>87.03</td>
      <td>0</td>
      <td>0</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-7</td>
      <td>113</td>
      <td>41061.95</td>
      <td>24.483867</td>
      <td>9.2</td>
      <td>225.10</td>
      <td>2.151365</td>
      <td>4.7</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>45000.0</td>
      <td>715</td>
      <td>69.4</td>
      <td>9.2</td>
      <td>87.03</td>
      <td>0</td>
      <td>0</td>
      <td>26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>-7</td>
      <td>113</td>
      <td>40804.42</td>
      <td>24.626795</td>
      <td>9.2</td>
      <td>222.39</td>
      <td>2.361722</td>
      <td>4.4</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>45000.0</td>
      <td>715</td>
      <td>69.4</td>
      <td>9.2</td>
      <td>87.03</td>
      <td>0</td>
      <td>0</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>-7</td>
      <td>113</td>
      <td>40483.89</td>
      <td>24.735883</td>
      <td>9.2</td>
      <td>219.67</td>
      <td>1.229172</td>
      <td>4.6</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>45000.0</td>
      <td>715</td>
      <td>69.4</td>
      <td>9.2</td>
      <td>87.03</td>
      <td>0</td>
      <td>0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>-7</td>
      <td>113</td>
      <td>40367.06</td>
      <td>24.925476</td>
      <td>9.2</td>
      <td>217.37</td>
      <td>1.692969</td>
      <td>4.5</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>45000.0</td>
      <td>715</td>
      <td>69.4</td>
      <td>9.2</td>
      <td>87.03</td>
      <td>0</td>
      <td>0</td>
      <td>29</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



### Fitting the model


```python
from lifelines import CoxTimeVaryingFitter

cox_tv = CoxTimeVaryingFitter()
cox_tv.fit(data_cox_tv, id_col="id", event_col="default_time", start_col="start", stop_col="stop", show_progress=True, step_size=0.1)
```

    Iteration 30: norm_delta = 0.00000, step_size = 1.00000, ll = -179884.48600, newton_decrement = 0.00000, seconds_since_start = 220.1Convergence completed after 30 iterations.
    




    <lifelines.CoxTimeVaryingFitter: fitted with 622489 periods, 50000 subjects, 15158 events>




```python
cox_tv.print_summary()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.CoxTimeVaryingFitter</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'default_time'</td>
    </tr>
    <tr>
      <th>number of subjects</th>
      <td>50000</td>
    </tr>
    <tr>
      <th>number of periods</th>
      <td>622489</td>
    </tr>
    <tr>
      <th>number of events</th>
      <td>15158</td>
    </tr>
    <tr>
      <th>partial log-likelihood</th>
      <td>-179884.49</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2020-01-22 10:28:32 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coef</th>
      <th>exp(coef)</th>
      <th>se(coef)</th>
      <th>coef lower 95%</th>
      <th>coef upper 95%</th>
      <th>exp(coef) lower 95%</th>
      <th>exp(coef) upper 95%</th>
      <th>z</th>
      <th>p</th>
      <th>-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>origination_time</th>
      <td>-0.03</td>
      <td>0.97</td>
      <td>0.00</td>
      <td>-0.03</td>
      <td>-0.02</td>
      <td>0.97</td>
      <td>0.98</td>
      <td>-10.33</td>
      <td>&lt;0.005</td>
      <td>80.63</td>
    </tr>
    <tr>
      <th>maturity_time</th>
      <td>0.01</td>
      <td>1.01</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>1.01</td>
      <td>1.01</td>
      <td>13.98</td>
      <td>&lt;0.005</td>
      <td>145.10</td>
    </tr>
    <tr>
      <th>balance_time</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>20.95</td>
      <td>&lt;0.005</td>
      <td>321.22</td>
    </tr>
    <tr>
      <th>LTV_time</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>10.10</td>
      <td>&lt;0.005</td>
      <td>77.31</td>
    </tr>
    <tr>
      <th>interest_rate_time</th>
      <td>0.24</td>
      <td>1.27</td>
      <td>0.00</td>
      <td>0.23</td>
      <td>0.24</td>
      <td>1.26</td>
      <td>1.27</td>
      <td>81.70</td>
      <td>&lt;0.005</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>house_price_index_time</th>
      <td>0.03</td>
      <td>1.03</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>1.03</td>
      <td>1.03</td>
      <td>30.26</td>
      <td>&lt;0.005</td>
      <td>665.89</td>
    </tr>
    <tr>
      <th>gdp_time</th>
      <td>-0.32</td>
      <td>0.73</td>
      <td>0.01</td>
      <td>-0.33</td>
      <td>-0.31</td>
      <td>0.72</td>
      <td>0.73</td>
      <td>-59.46</td>
      <td>&lt;0.005</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>unemployment_rate_time</th>
      <td>-0.25</td>
      <td>0.78</td>
      <td>0.01</td>
      <td>-0.27</td>
      <td>-0.23</td>
      <td>0.76</td>
      <td>0.79</td>
      <td>-24.61</td>
      <td>&lt;0.005</td>
      <td>441.79</td>
    </tr>
    <tr>
      <th>real_estate_condominium</th>
      <td>0.20</td>
      <td>1.22</td>
      <td>0.04</td>
      <td>0.13</td>
      <td>0.27</td>
      <td>1.14</td>
      <td>1.32</td>
      <td>5.49</td>
      <td>&lt;0.005</td>
      <td>24.61</td>
    </tr>
    <tr>
      <th>real_estate_planned_urban_dev</th>
      <td>0.17</td>
      <td>1.19</td>
      <td>0.03</td>
      <td>0.11</td>
      <td>0.23</td>
      <td>1.12</td>
      <td>1.26</td>
      <td>5.59</td>
      <td>&lt;0.005</td>
      <td>25.35</td>
    </tr>
    <tr>
      <th>real_estate_single_family_home</th>
      <td>0.07</td>
      <td>1.07</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.11</td>
      <td>1.02</td>
      <td>1.11</td>
      <td>3.00</td>
      <td>&lt;0.005</td>
      <td>8.51</td>
    </tr>
    <tr>
      <th>investor_orig_time</th>
      <td>0.10</td>
      <td>1.10</td>
      <td>0.03</td>
      <td>0.05</td>
      <td>0.15</td>
      <td>1.05</td>
      <td>1.16</td>
      <td>3.81</td>
      <td>&lt;0.005</td>
      <td>12.82</td>
    </tr>
    <tr>
      <th>balance_orig_time</th>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>-18.64</td>
      <td>&lt;0.005</td>
      <td>255.10</td>
    </tr>
    <tr>
      <th>FICO_orig_time</th>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>-24.93</td>
      <td>&lt;0.005</td>
      <td>453.21</td>
    </tr>
    <tr>
      <th>LTV_orig_time</th>
      <td>0.01</td>
      <td>1.01</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>1.01</td>
      <td>1.01</td>
      <td>10.53</td>
      <td>&lt;0.005</td>
      <td>83.72</td>
    </tr>
    <tr>
      <th>interest_rate_orig_time</th>
      <td>-0.03</td>
      <td>0.97</td>
      <td>0.00</td>
      <td>-0.04</td>
      <td>-0.03</td>
      <td>0.96</td>
      <td>0.97</td>
      <td>-13.59</td>
      <td>&lt;0.005</td>
      <td>137.35</td>
    </tr>
    <tr>
      <th>house_price_index_orig_time</th>
      <td>0.01</td>
      <td>1.01</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>1.01</td>
      <td>1.01</td>
      <td>13.90</td>
      <td>&lt;0.005</td>
      <td>143.59</td>
    </tr>
  </tbody>
</table><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>Log-likelihood ratio test</th>
      <td>18923.94 on 17 df, -log2(p)=inf</td>
    </tr>
  </tbody>
</table>
</div>


### Filter down to just censored subjects to predict remaining survival


```python
censored_subjects_tv = data_cox_tv.loc[~data_cox_tv['default_time'].astype(bool)]
censored_subjects_last_obs_tv = censored_subjects_tv['stop']
```


```python
cox_tv.predict_partial_hazard(censored_subjects_tv).head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.267893</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.774304</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.578277</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.238890</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.674618</td>
    </tr>
  </tbody>
</table>
</div>



### Plotting the coefficients


```python
cox_tv.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249b4570c50>




![png](output_133_1.png)



```python
cox_tv.baseline_survival_[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>baseline survival</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.999998</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.999995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.999971</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.999959</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.999948</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.999932</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.999924</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.999911</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.999903</td>
    </tr>
  </tbody>
</table>
</div>



### The estimated coefficients

Changed in version 0.22.0: use to be .hazards_


```python
cox_tv.params_
```




    origination_time                 -0.026377
    maturity_time                     0.008612
    balance_time                      0.000007
    LTV_time                          0.003312
    interest_rate_time                0.236499
    house_price_index_time            0.029767
    gdp_time                         -0.319236
    unemployment_rate_time           -0.252926
    real_estate_condominium           0.202032
    real_estate_planned_urban_dev     0.172676
    real_estate_single_family_home    0.065743
    investor_orig_time                0.097674
    balance_orig_time                -0.000007
    FICO_orig_time                   -0.003158
    LTV_orig_time                     0.009219
    interest_rate_orig_time          -0.034306
    house_price_index_orig_time       0.008454
    dtype: float64



### Hazard ratios

The exp(coefficients)


```python
cox_tv.hazard_ratios_
```




    origination_time                  0.973968
    maturity_time                     1.008649
    balance_time                      1.000007
    LTV_time                          1.003318
    interest_rate_time                1.266806
    house_price_index_time            1.030214
    gdp_time                          0.726704
    unemployment_rate_time            0.776525
    real_estate_condominium           1.223887
    real_estate_planned_urban_dev     1.188481
    real_estate_single_family_home    1.067952
    investor_orig_time                1.102604
    balance_orig_time                 0.999993
    FICO_orig_time                    0.996847
    LTV_orig_time                     1.009261
    interest_rate_orig_time           0.966276
    house_price_index_orig_time       1.008490
    Name: exp(coef), dtype: float64



### Concordance index


```python
cindex_tv = concordance_index(data_cox_tv['stop'], -cox_tv.predict_partial_hazard(data_cox_tv), data_cox_tv['default_time'])
```


```python
cindex_tv
```




    0.8169598317017039



---

## 4.d Random Survival Forests<a class="anchor" id="four_d"></a>


```python
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
```


```python
rstate = 124
```


```python
def get_x_y_survival(dataset, col_event, col_time, val_outcome):
    if col_event is None or col_time is None:
        y = None
        x_frame = dataset
    else:
        y = np.empty(dtype=[(col_event, np.bool), (col_time, np.float64)],
                        shape=dataset.shape[0])
        y[col_event] = (dataset[col_event] == val_outcome).values
        y[col_time] = dataset[col_time].values

        x_frame = dataset.drop([col_event, col_time], axis=1)

    return x_frame, y
```


```python
X_rf, y_rf = get_x_y_survival(data_cox, 'default_time', 'total_obs_time', 1)
```


```python
X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf, test_size=0.25, random_state=rstate)
```


```python
rsf = RandomSurvivalForest(n_estimators=50,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           max_features="sqrt",
                           n_jobs=-1,
                           random_state=rstate,
                          verbose=1)
```


```python
rsf.fit(X_rf_train, y_rf_train)
```

    [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  6.6min
    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:  7.5min finished
    




    RandomSurvivalForest(bootstrap=True, max_depth=None, max_features='sqrt',
                         max_leaf_nodes=None, min_samples_leaf=15,
                         min_samples_split=10, min_weight_fraction_leaf=0.0,
                         n_estimators=50, n_jobs=-1, oob_score=False,
                         random_state=124, verbose=1, warm_start=False)




```python
rsf.score(X_rf_test, y_rf_test)
```

    [Parallel(n_jobs=4)]: Using backend ThreadingBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=4)]: Done  50 out of  50 | elapsed:    0.1s finished
    




    0.7970134777974758



---

# 5. Deep Learning in Survival Analysis<a class="anchor" id="five"></a>

## 5.a DeepSurv<a class="anchor" id="five_a"></a>


```python
# For preprocessing
import sklearn_pandas
from sklearn_pandas import DataFrameMapper 

from sklearn.model_selection import train_test_split

import torch
import torchtuples as tt

from pycox.evaluation import EvalSurv
from pycox.models import CoxPH, CoxCC, CoxTime

from pycox.models.cox_time import MLPVanillaCoxTime
```


```python
# We also set some seeds to make this reproducable.
# Note that on gpu, there is still some randomness.
np.random.seed(1234)
_ = torch.manual_seed(123)
```

### Data preprocessing


```python
data_ds = data_cox.copy()
```


```python
#data_ds = data_ds.drop('id', axis = 1)
```


```python
df_train = data_ds.copy()
```


```python
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)
```


```python
cols_stand = ['balance_time', 'LTV_time', 'origination_time', 'maturity_time',
       'interest_rate_time', 'house_price_index_time', 'gdp_time', 'unemployment_rate_time',
       'balance_orig_time', 'FICO_orig_time',
       'LTV_orig_time', 'interest_rate_orig_time', 'house_price_index_orig_time']

cols_leave = ['investor_orig_time', 'real_estate_condominium',
       'real_estate_planned_urban_dev', 'real_estate_single_family_home', 'total_obs_time', 'default_time']

#standardize = [([col], StandardScaler) for col in cols_stand]

standardize = [([col], StandardScaler()) for col in cols_stand]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')
```


```python
print(x_train.shape, x_test.shape, x_val.shape)
```

    (32000, 19) (10000, 19) (8000, 19)
    


```python
labtrans = CoxTime.label_transform()
```


```python
get_target = lambda df: (df['total_obs_time'].values, df['default_time'].values)
y_train = labtrans.fit_transform(*get_target(df_train))
y_val = labtrans.transform(*get_target(df_val))
durations_test, events_test = get_target(df_test)
val = tt.tuplefy(x_val, y_val)
```


```python
'''
get_target = lambda df: (df['total_obs_time'].values, df['default_time'].values)
y_train = get_target(df_train)
y_val = get_target(df_val)
durations_test, events_test = get_target(df_test)
val = x_val, y_val
'''
```

### Simple NN


```python
from torch.nn import Dropout, Linear, Sequential, ReLU, SELU, BatchNorm1d
from torch.optim import SGD, Adam
```


```python
n_nodes = 256
```


```python
in_features = x_train.shape[1]
num_nodes = [n_nodes, n_nodes, n_nodes, n_nodes]
out_features = 1
batch_norm = True
dropout = 0.4
output_bias = False
```


```python
net_ds = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout, output_bias=output_bias)
```


```python
'''
network_ds = Sequential(
    Linear(in_features, n_nodes),
    ReLU(),
    BatchNorm1d(n_nodes),
    Dropout(0.2),
    
    Linear(n_nodes, n_nodes),
    ReLU(),
    BatchNorm1d(n_nodes),
    Dropout(0.2),
        
    Linear(n_nodes, n_nodes),
    ReLU(),
    BatchNorm1d(n_nodes),
    Dropout(0.2),
    Linear(n_nodes, out_features)
)
'''
```




    '\nnetwork_ds = Sequential(\n    Linear(in_features, n_nodes),\n    ReLU(),\n    BatchNorm1d(n_nodes),\n    Dropout(0.2),\n    \n    Linear(n_nodes, n_nodes),\n    ReLU(),\n    BatchNorm1d(n_nodes),\n    Dropout(0.2),\n        \n    Linear(n_nodes, n_nodes),\n    ReLU(),\n    BatchNorm1d(n_nodes),\n    Dropout(0.2),\n    Linear(n_nodes, out_features)\n)\n'




```python
net_ds
```




    MLPVanilla(
      (net): Sequential(
        (0): DenseVanillaBlock(
          (linear): Linear(in_features=19, out_features=256, bias=True)
          (activation): ReLU()
          (batch_norm): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (dropout): Dropout(p=0.4, inplace=False)
        )
        (1): DenseVanillaBlock(
          (linear): Linear(in_features=256, out_features=256, bias=True)
          (activation): ReLU()
          (batch_norm): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (dropout): Dropout(p=0.4, inplace=False)
        )
        (2): DenseVanillaBlock(
          (linear): Linear(in_features=256, out_features=256, bias=True)
          (activation): ReLU()
          (batch_norm): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (dropout): Dropout(p=0.4, inplace=False)
        )
        (3): DenseVanillaBlock(
          (linear): Linear(in_features=256, out_features=256, bias=True)
          (activation): ReLU()
          (batch_norm): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (dropout): Dropout(p=0.4, inplace=False)
        )
        (4): Linear(in_features=256, out_features=1, bias=False)
      )
    )




```python
model_deepsurv = CoxPH(net_ds, tt.optim.Adam)
```

### Training the model


```python
batch_size = 128
```


```python
lrfinder = model_deepsurv.lr_finder(x_train, y_train, batch_size, tolerance=10)
_ = lrfinder.plot()
```


![png](output_179_0.png)



```python
lrfinder.get_best_lr()
```




    0.020092330025650584




```python
model_deepsurv.optimizer.set_lr(0.001)
```


```python
model_deepsurv.optimizer.param_groups[0]['lr']
```




    0.001




```python
epochs = 512

callbacks = [tt.callbacks.EarlyStopping()]
verbose = True
```


```python
%%time
log = model_deepsurv.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)
```

    0:	[4s / 4s],		train_loss: 1.9863,	val_loss: 1.3404
    1:	[4s / 9s],		train_loss: 1.5710,	val_loss: 1.2413
    2:	[4s / 13s],		train_loss: 1.4730,	val_loss: 1.2831
    3:	[4s / 17s],		train_loss: 1.4188,	val_loss: 1.1973
    4:	[4s / 22s],		train_loss: 1.3922,	val_loss: 1.1636
    5:	[4s / 27s],		train_loss: 1.3742,	val_loss: 1.1751
    6:	[4s / 31s],		train_loss: 1.3561,	val_loss: 1.1691
    7:	[4s / 36s],		train_loss: 1.3360,	val_loss: 1.1749
    8:	[4s / 41s],		train_loss: 1.3199,	val_loss: 1.1696
    9:	[4s / 45s],		train_loss: 1.3161,	val_loss: 1.1587
    10:	[4s / 50s],		train_loss: 1.3026,	val_loss: 1.1544
    11:	[4s / 55s],		train_loss: 1.3060,	val_loss: 1.1409
    12:	[4s / 59s],		train_loss: 1.2941,	val_loss: 1.1413
    13:	[4s / 1m:4s],		train_loss: 1.2915,	val_loss: 1.1594
    14:	[4s / 1m:8s],		train_loss: 1.2798,	val_loss: 1.1357
    15:	[4s / 1m:13s],		train_loss: 1.2804,	val_loss: 1.1394
    16:	[4s / 1m:17s],		train_loss: 1.2777,	val_loss: 1.1440
    17:	[4s / 1m:22s],		train_loss: 1.2713,	val_loss: 1.1497
    18:	[4s / 1m:27s],		train_loss: 1.2602,	val_loss: 1.1403
    19:	[4s / 1m:31s],		train_loss: 1.2611,	val_loss: 1.1359
    20:	[4s / 1m:36s],		train_loss: 1.2645,	val_loss: 1.1457
    21:	[4s / 1m:41s],		train_loss: 1.2601,	val_loss: 1.1320
    22:	[4s / 1m:45s],		train_loss: 1.2609,	val_loss: 1.1450
    23:	[4s / 1m:49s],		train_loss: 1.2564,	val_loss: 1.1264
    24:	[4s / 1m:54s],		train_loss: 1.2593,	val_loss: 1.1339
    25:	[4s / 1m:58s],		train_loss: 1.2522,	val_loss: 1.1760
    26:	[4s / 2m:3s],		train_loss: 1.2563,	val_loss: 1.1305
    27:	[4s / 2m:7s],		train_loss: 1.2506,	val_loss: 1.1296
    28:	[4s / 2m:11s],		train_loss: 1.2461,	val_loss: 1.1272
    29:	[4s / 2m:16s],		train_loss: 1.2465,	val_loss: 1.1299
    30:	[4s / 2m:20s],		train_loss: 1.2466,	val_loss: 1.1478
    31:	[4s / 2m:24s],		train_loss: 1.2438,	val_loss: 1.1376
    32:	[5s / 2m:30s],		train_loss: 1.2423,	val_loss: 1.1320
    33:	[4s / 2m:34s],		train_loss: 1.2422,	val_loss: 1.1390
    Wall time: 2min 34s
    


```python
_ = log.plot()
```


![png](output_185_0.png)



```python
model_deepsurv.partial_log_likelihood(*val).mean()
```




    -5.242654




```python
model_deepsurv.score_in_batches(val)
```




    {'loss': 4.915094375610352}



### Prediction


```python
_ = model_deepsurv.compute_baseline_hazards()
```


```python
deepsurv = model_deepsurv.predict_surv_df(x_test)
```


```python
deepsurv.iloc[:, :5].plot()
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')
```


![png](output_191_0.png)



```python
model_deepsurv.baseline_hazards_.head()
```




    duration
    -0.984273    0.000192
    -0.898189    0.000487
    -0.812104    0.001240
    -0.726020    0.003143
    -0.639936    0.009442
    Name: baseline_hazards, dtype: float32



### Evaluation Metrics


```python
ev = EvalSurv(deepsurv, durations_test, events_test, censor_surv='km')
```


```python
ev.concordance_td()
```




    0.8424150843984598




```python
time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
_ = ev.brier_score(time_grid).plot()
```


![png](output_196_0.png)



```python
ev.integrated_brier_score(time_grid)
```




    0.11487176920079373




```python
ev.integrated_nbll(time_grid)
```




    1.6275194724001085



---

## 5.b DeepHit<a class="anchor" id="five_b"></a> 


```python
data_sr = df.copy()
data_cr = df.copy()
```


```python
data_sr.insert(1, 'event',data_sr['default_time'])
data_sr.insert(2, 'duration',data_sr['total_obs_time'])
data_sr = data_sr.drop(['time','first_time','default_time','payoff_time','status_time','time_max','total_obs_time'],axis=1)

data_cr.insert(1, 'event',data_cr['status_time'])
data_cr.insert(2, 'duration',data_cr['total_obs_time'])
data_cr = data_cr.drop(['time','first_time','default_time','payoff_time','time_max','total_obs_time','status_time'],axis=1)
```


```python
data_cr.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>event</th>
      <th>duration</th>
      <th>orig_time</th>
      <th>mat_time</th>
      <th>balance_time</th>
      <th>LTV_time</th>
      <th>interest_rate_time</th>
      <th>hpi_time</th>
      <th>gdp_time</th>
      <th>uer_time</th>
      <th>REtype_CO_orig_time</th>
      <th>REtype_PU_orig_time</th>
      <th>REtype_SF_orig_time</th>
      <th>investor_orig_time</th>
      <th>balance_orig_time</th>
      <th>FICO_orig_time</th>
      <th>LTV_orig_time</th>
      <th>Interest_Rate_orig_time</th>
      <th>hpi_orig_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>1</td>
      <td>24</td>
      <td>-7</td>
      <td>113</td>
      <td>29087.21</td>
      <td>26.658065</td>
      <td>9.200</td>
      <td>146.45</td>
      <td>2.715903</td>
      <td>8.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>45000.0</td>
      <td>715</td>
      <td>69.4</td>
      <td>9.200</td>
      <td>87.03</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>18</td>
      <td>138</td>
      <td>105654.77</td>
      <td>65.469851</td>
      <td>7.680</td>
      <td>225.10</td>
      <td>2.151365</td>
      <td>4.7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>107200.0</td>
      <td>558</td>
      <td>80.0</td>
      <td>7.680</td>
      <td>186.91</td>
    </tr>
    <tr>
      <th>30</th>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>-6</td>
      <td>114</td>
      <td>44378.60</td>
      <td>31.459735</td>
      <td>11.375</td>
      <td>217.37</td>
      <td>1.692969</td>
      <td>4.5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>48600.0</td>
      <td>680</td>
      <td>83.6</td>
      <td>8.750</td>
      <td>89.58</td>
    </tr>
    <tr>
      <th>65</th>
      <td>4</td>
      <td>0</td>
      <td>36</td>
      <td>-2</td>
      <td>119</td>
      <td>52686.35</td>
      <td>34.898842</td>
      <td>10.500</td>
      <td>189.82</td>
      <td>2.836358</td>
      <td>5.7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>63750.0</td>
      <td>587</td>
      <td>81.8</td>
      <td>10.500</td>
      <td>97.99</td>
    </tr>
    <tr>
      <th>68</th>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>18</td>
      <td>138</td>
      <td>52100.71</td>
      <td>66.346343</td>
      <td>9.155</td>
      <td>222.39</td>
      <td>2.361722</td>
      <td>4.4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>52800.0</td>
      <td>527</td>
      <td>80.0</td>
      <td>9.155</td>
      <td>186.91</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_sr.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>event</th>
      <th>duration</th>
      <th>orig_time</th>
      <th>mat_time</th>
      <th>balance_time</th>
      <th>LTV_time</th>
      <th>interest_rate_time</th>
      <th>hpi_time</th>
      <th>gdp_time</th>
      <th>uer_time</th>
      <th>REtype_CO_orig_time</th>
      <th>REtype_PU_orig_time</th>
      <th>REtype_SF_orig_time</th>
      <th>investor_orig_time</th>
      <th>balance_orig_time</th>
      <th>FICO_orig_time</th>
      <th>LTV_orig_time</th>
      <th>Interest_Rate_orig_time</th>
      <th>hpi_orig_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>1</td>
      <td>24</td>
      <td>-7</td>
      <td>113</td>
      <td>29087.21</td>
      <td>26.658065</td>
      <td>9.200</td>
      <td>146.45</td>
      <td>2.715903</td>
      <td>8.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>45000.0</td>
      <td>715</td>
      <td>69.4</td>
      <td>9.200</td>
      <td>87.03</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>18</td>
      <td>138</td>
      <td>105654.77</td>
      <td>65.469851</td>
      <td>7.680</td>
      <td>225.10</td>
      <td>2.151365</td>
      <td>4.7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>107200.0</td>
      <td>558</td>
      <td>80.0</td>
      <td>7.680</td>
      <td>186.91</td>
    </tr>
    <tr>
      <th>30</th>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>-6</td>
      <td>114</td>
      <td>44378.60</td>
      <td>31.459735</td>
      <td>11.375</td>
      <td>217.37</td>
      <td>1.692969</td>
      <td>4.5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>48600.0</td>
      <td>680</td>
      <td>83.6</td>
      <td>8.750</td>
      <td>89.58</td>
    </tr>
    <tr>
      <th>65</th>
      <td>4</td>
      <td>0</td>
      <td>36</td>
      <td>-2</td>
      <td>119</td>
      <td>52686.35</td>
      <td>34.898842</td>
      <td>10.500</td>
      <td>189.82</td>
      <td>2.836358</td>
      <td>5.7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>63750.0</td>
      <td>587</td>
      <td>81.8</td>
      <td>10.500</td>
      <td>97.99</td>
    </tr>
    <tr>
      <th>68</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>18</td>
      <td>138</td>
      <td>52100.71</td>
      <td>66.346343</td>
      <td>9.155</td>
      <td>222.39</td>
      <td>2.361722</td>
      <td>4.4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>52800.0</td>
      <td>527</td>
      <td>80.0</td>
      <td>9.155</td>
      <td>186.91</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_sr['event'].value_counts()
```




    0    34846
    1    15154
    Name: event, dtype: int64




```python
default = data_sr.copy()
censored = data_sr.copy()
```


```python
default.drop(default.loc[default['event']==0].index, inplace=True)
```


```python
censored.drop(censored.loc[censored['event']==1].index, inplace=True)
```


```python
print('default')
print("medium time:", default.duration.mean())
print("max time:", default.duration.max())
print("min time:", default.duration.min())
print(' ')
print('censored')
print("medium time:", censored.duration.mean())
print("max time:", censored.duration.max())
print("min time:", censored.duration.min())
```

    default
    medium time: 10.554111125775373
    max time: 44
    min time: 1
     
    censored
    medium time: 13.309849050106182
    max time: 60
    min time: 1
    


```python
from class_DeepHit import Model_DeepHit, log, div
import import_data
from import_data import f_get_Normalization, f_get_fc_mask2, f_get_fc_mask3
from get_main import f_get_minibatch 
from main_RandomSearch import save_logging, load_logging , get_random_hyperparameters
import numpy as np
import tensorflow as tf
import random
import time, datetime, os
import get_main

import import_data as impt
```


```python
from tensorflow.contrib.layers import fully_connected as FC_Net
from sklearn.model_selection import train_test_split

import utils_network as utils
from termcolor import colored
from utils_eval import c_index, brier_score, weighted_c_index, weighted_brier_score
from summarize_results import load_logging
from termcolor import colored
from utils_eval import c_index, brier_score, weighted_c_index, weighted_brier_score
```


```python
# DeepHit single risk
norm_mode='standard'
label              = np.asarray(data_sr[['event']])
time               = np.asarray(data_sr[['duration']])
dhdata             = np.asarray(data_sr.iloc[:,1:18])
dhdata             = f_get_Normalization(dhdata, norm_mode)

num_Category       = int(np.max(time) * 1.2)  #to have enough time-horizon
num_Event          = int(len(np.unique(label)) - 1) #only count the number of events (do not count censoring as an event)

x_dim              = np.shape(dhdata)[1]

mask1              = f_get_fc_mask2(time, label, num_Event, num_Category)
mask2              = f_get_fc_mask3(time, -1, num_Category)

DIM                = (x_dim)
DATA               = (dhdata, time, label)
MASK               = (mask1, mask2)

_EPSILON = 1e-08
seed = 1234
```


```python
# DeepHit competing risks
norm_mode='standard'
label              = np.asarray(data_cr[['event']])
time               = np.asarray(data_cr[['duration']])
dhdata_cr          = np.asarray(data_cr.iloc[:,1:18])
dhdata_cr          = f_get_Normalization(dhdata, norm_mode)

num_Category       = int(np.max(time) * 1.2)  #to have enough time-horizon
num_Event          = int(len(np.unique(label)) - 1) #only count the number of events (do not count censoring as an event)

x_dim              = np.shape(dhdata)[1]

mask1              = f_get_fc_mask2(time, label, num_Event, num_Category)
mask2              = f_get_fc_mask3(time, -1, num_Category)

DIM                = (x_dim)
DATA               = (dhdata_cr, time, label)
MASK               = (mask1, mask2)

_EPSILON = 1e-08
seed = 1234
```


```python
OUT_ITERATION               = 1
RS_ITERATION                = 2
data_mode = 'mortgage'
eval_times = [3,7,17]
```


```python
if num_Event > 1:
    out_path      = data_mode + '/results_cr/'
    print('DeepHit with competing risks')
else: 
    out_path      = data_mode + '/results/'
    print('DeepHit with single risk')
```

    DeepHit with competing risks
    


```python
for itr in range(OUT_ITERATION):
    
    if not os.path.exists(out_path + '/itr_' + str(itr) + '/'):
        os.makedirs(out_path + '/itr_' + str(itr) + '/')

    max_valid = 0.
    log_name = out_path + '/itr_' + str(itr) + '/hyperparameters_log.txt'

    for r_itr in range(RS_ITERATION):
        print('OUTER_ITERATION: ' + str(itr))
        print('Random search... itr: ' + str(r_itr))
        new_parser = get_random_hyperparameters(out_path)
        print(new_parser)

        # get validation performance given the hyperparameters
        tmp_max = get_main.get_valid_performance(DATA, MASK, new_parser, itr, eval_times, MAX_VALUE=max_valid)

        if tmp_max > max_valid:
            max_valid = tmp_max
            max_parser = new_parser
            save_logging(max_parser, log_name)  #save the hyperparameters if this provides the maximum validation performance

        print('Current best: ' + str(max_valid))
```

    OUTER_ITERATION: 0
    Random search... itr: 0
    {'mb_size': 32, 'iteration': 50000, 'keep_prob': 0.6, 'lr_train': 0.0001, 'h_dim_shared': 200, 'h_dim_CS': 200, 'num_layers_shared': 1, 'num_layers_CS': 2, 'active_fn': 'relu', 'alpha': 1.0, 'beta': 0.1, 'gamma': 0, 'out_path': 'mortgage/results/'}
    mortgage/results//itr_0 (a:1.0 b:0.1 c:0)
    WARNING:tensorflow:From /Users/lauraloschmann/anaconda3/envs/Python1/lib/python3.7/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    WARNING:tensorflow:From /Users/lauraloschmann/Documents/Humboldt/Master/3. Semester/BIS/utils_network.py:94: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
    WARNING:tensorflow:From /Users/lauraloschmann/Documents/Humboldt/Master/3. Semester/BIS/class_DeepHit.py:120: calling reduce_sum_v1 (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
    Instructions for updating:
    keep_dims is deprecated, use keepdims instead
    WARNING:tensorflow:From /Users/lauraloschmann/anaconda3/envs/Python1/lib/python3.7/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    MAIN TRAINING ...
    EVALUATION TIMES: [3, 7, 17]
    || ITR: 1000 | Loss: [1m[33m1.6306[0m
    updated.... average c-index = 0.8502
    || ITR: 2000 | Loss: [1m[33m1.4631[0m
    updated.... average c-index = 0.8690
    || ITR: 3000 | Loss: [1m[33m1.3790[0m
    updated.... average c-index = 0.8747
    || ITR: 4000 | Loss: [1m[33m1.3652[0m
    updated.... average c-index = 0.8826
    || ITR: 5000 | Loss: [1m[33m1.2966[0m
    updated.... average c-index = 0.8890
    || ITR: 6000 | Loss: [1m[33m1.2899[0m
    updated.... average c-index = 0.8917
    || ITR: 7000 | Loss: [1m[33m1.2885[0m
    updated.... average c-index = 0.8951
    || ITR: 8000 | Loss: [1m[33m1.2428[0m
    updated.... average c-index = 0.8962
    || ITR: 9000 | Loss: [1m[33m1.2611[0m
    updated.... average c-index = 0.8990
    || ITR: 10000 | Loss: [1m[33m1.2328[0m
    updated.... average c-index = 0.9028
    || ITR: 11000 | Loss: [1m[33m1.2214[0m
    || ITR: 12000 | Loss: [1m[33m1.2006[0m
    updated.... average c-index = 0.9040
    || ITR: 13000 | Loss: [1m[33m1.1918[0m
    || ITR: 14000 | Loss: [1m[33m1.1835[0m
    || ITR: 15000 | Loss: [1m[33m1.1769[0m
    updated.... average c-index = 0.9048
    || ITR: 16000 | Loss: [1m[33m1.1899[0m
    updated.... average c-index = 0.9068
    || ITR: 17000 | Loss: [1m[33m1.1573[0m
    || ITR: 18000 | Loss: [1m[33m1.1658[0m
    || ITR: 19000 | Loss: [1m[33m1.1450[0m
    || ITR: 20000 | Loss: [1m[33m1.1472[0m
    updated.... average c-index = 0.9080
    || ITR: 21000 | Loss: [1m[33m1.1457[0m
    || ITR: 22000 | Loss: [1m[33m1.0996[0m
    || ITR: 23000 | Loss: [1m[33m1.1214[0m
    || ITR: 24000 | Loss: [1m[33m1.1112[0m
    updated.... average c-index = 0.9104
    || ITR: 25000 | Loss: [1m[33m1.1214[0m
    || ITR: 26000 | Loss: [1m[33m1.1069[0m
    || ITR: 27000 | Loss: [1m[33m1.1135[0m
    || ITR: 28000 | Loss: [1m[33m1.1252[0m
    updated.... average c-index = 0.9106
    || ITR: 29000 | Loss: [1m[33m1.0921[0m
    || ITR: 30000 | Loss: [1m[33m1.1054[0m
    updated.... average c-index = 0.9113
    || ITR: 31000 | Loss: [1m[33m1.1068[0m
    || ITR: 32000 | Loss: [1m[33m1.0989[0m
    || ITR: 33000 | Loss: [1m[33m1.0958[0m
    || ITR: 34000 | Loss: [1m[33m1.0806[0m
    updated.... average c-index = 0.9120
    || ITR: 35000 | Loss: [1m[33m1.0799[0m
    || ITR: 36000 | Loss: [1m[33m1.0912[0m
    || ITR: 37000 | Loss: [1m[33m1.1047[0m
    || ITR: 38000 | Loss: [1m[33m1.0715[0m
    || ITR: 39000 | Loss: [1m[33m1.0899[0m
    || ITR: 40000 | Loss: [1m[33m1.0671[0m
    updated.... average c-index = 0.9125
    || ITR: 41000 | Loss: [1m[33m1.0542[0m
    || ITR: 42000 | Loss: [1m[33m1.0695[0m
    || ITR: 43000 | Loss: [1m[33m1.0650[0m
    || ITR: 44000 | Loss: [1m[33m1.0803[0m
    updated.... average c-index = 0.9144
    || ITR: 45000 | Loss: [1m[33m1.0612[0m
    updated.... average c-index = 0.9149
    || ITR: 46000 | Loss: [1m[33m1.0631[0m
    || ITR: 47000 | Loss: [1m[33m1.0554[0m
    || ITR: 48000 | Loss: [1m[33m1.0703[0m
    || ITR: 49000 | Loss: [1m[33m1.0336[0m
    || ITR: 50000 | Loss: [1m[33m1.0739[0m
    Current best: 0.914880908002149
    OUTER_ITERATION: 0
    Random search... itr: 1
    {'mb_size': 64, 'iteration': 50000, 'keep_prob': 0.6, 'lr_train': 0.0001, 'h_dim_shared': 300, 'h_dim_CS': 50, 'num_layers_shared': 1, 'num_layers_CS': 3, 'active_fn': 'relu', 'alpha': 1.0, 'beta': 0.5, 'gamma': 0, 'out_path': 'mortgage/results/'}
    mortgage/results//itr_0 (a:1.0 b:0.5 c:0)
    MAIN TRAINING ...
    EVALUATION TIMES: [3, 7, 17]
    || ITR: 1000 | Loss: [1m[33m5.0772[0m
    updated.... average c-index = 0.8609
    || ITR: 2000 | Loss: [1m[33m4.2156[0m
    updated.... average c-index = 0.8817
    || ITR: 3000 | Loss: [1m[33m3.8557[0m
    updated.... average c-index = 0.8905
    || ITR: 4000 | Loss: [1m[33m3.6382[0m
    updated.... average c-index = 0.8955
    || ITR: 5000 | Loss: [1m[33m3.5050[0m
    updated.... average c-index = 0.9024
    || ITR: 6000 | Loss: [1m[33m3.3800[0m
    updated.... average c-index = 0.9049
    || ITR: 7000 | Loss: [1m[33m3.2070[0m
    updated.... average c-index = 0.9090
    || ITR: 8000 | Loss: [1m[33m3.1265[0m
    updated.... average c-index = 0.9111
    || ITR: 9000 | Loss: [1m[33m3.1534[0m
    updated.... average c-index = 0.9118
    || ITR: 10000 | Loss: [1m[33m3.0722[0m
    updated.... average c-index = 0.9149
    || ITR: 11000 | Loss: [1m[33m3.0015[0m
    || ITR: 12000 | Loss: [1m[33m2.9640[0m
    updated.... average c-index = 0.9176
    || ITR: 13000 | Loss: [1m[33m2.9627[0m
    || ITR: 14000 | Loss: [1m[33m2.8916[0m
    || ITR: 15000 | Loss: [1m[33m2.8975[0m
    || ITR: 16000 | Loss: [1m[33m2.8762[0m
    updated.... average c-index = 0.9200
    || ITR: 17000 | Loss: [1m[33m2.8541[0m
    updated.... average c-index = 0.9207
    || ITR: 18000 | Loss: [1m[33m2.8124[0m
    updated.... average c-index = 0.9207
    || ITR: 19000 | Loss: [1m[33m2.7737[0m
    updated.... average c-index = 0.9210
    || ITR: 20000 | Loss: [1m[33m2.7579[0m
    updated.... average c-index = 0.9217
    || ITR: 21000 | Loss: [1m[33m2.7374[0m
    updated.... average c-index = 0.9235
    || ITR: 22000 | Loss: [1m[33m2.7044[0m
    || ITR: 23000 | Loss: [1m[33m2.6650[0m
    || ITR: 24000 | Loss: [1m[33m2.6697[0m
    || ITR: 25000 | Loss: [1m[33m2.7111[0m
    || ITR: 26000 | Loss: [1m[33m2.6308[0m
    updated.... average c-index = 0.9239
    || ITR: 27000 | Loss: [1m[33m2.5832[0m
    || ITR: 28000 | Loss: [1m[33m2.6042[0m
    || ITR: 29000 | Loss: [1m[33m2.6173[0m
    || ITR: 30000 | Loss: [1m[33m2.5785[0m
    || ITR: 31000 | Loss: [1m[33m2.5171[0m
    || ITR: 32000 | Loss: [1m[33m2.5322[0m
    Current best: 0.9239002196511809
    OUTER_ITERATION: 0
    Random search... itr: 2
    {'mb_size': 128, 'iteration': 50000, 'keep_prob': 0.6, 'lr_train': 0.0001, 'h_dim_shared': 300, 'h_dim_CS': 100, 'num_layers_shared': 2, 'num_layers_CS': 5, 'active_fn': 'relu', 'alpha': 1.0, 'beta': 0.5, 'gamma': 0, 'out_path': 'mortgage/results/'}
    mortgage/results//itr_0 (a:1.0 b:0.5 c:0)
    MAIN TRAINING ...
    EVALUATION TIMES: [3, 7, 17]
    || ITR: 1000 | Loss: [1m[33m9.0685[0m
    updated.... average c-index = 0.8684
    || ITR: 2000 | Loss: [1m[33m7.0565[0m
    updated.... average c-index = 0.8864
    || ITR: 3000 | Loss: [1m[33m6.3867[0m
    updated.... average c-index = 0.8974
    || ITR: 4000 | Loss: [1m[33m5.8948[0m
    updated.... average c-index = 0.9026
    || ITR: 5000 | Loss: [1m[33m5.7028[0m
    updated.... average c-index = 0.9082
    || ITR: 6000 | Loss: [1m[33m5.3838[0m
    updated.... average c-index = 0.9101
    || ITR: 7000 | Loss: [1m[33m5.2414[0m
    updated.... average c-index = 0.9113
    || ITR: 8000 | Loss: [1m[33m5.0688[0m
    updated.... average c-index = 0.9116
    || ITR: 9000 | Loss: [1m[33m4.9177[0m
    || ITR: 10000 | Loss: [1m[33m4.8537[0m
    || ITR: 11000 | Loss: [1m[33m4.7001[0m
    || ITR: 12000 | Loss: [1m[33m4.6369[0m
    || ITR: 13000 | Loss: [1m[33m4.5727[0m
    || ITR: 14000 | Loss: [1m[33m4.4675[0m
    Current best: 0.9239002196511809
    OUTER_ITERATION: 0
    Random search... itr: 3
    {'mb_size': 64, 'iteration': 50000, 'keep_prob': 0.6, 'lr_train': 0.0001, 'h_dim_shared': 100, 'h_dim_CS': 200, 'num_layers_shared': 5, 'num_layers_CS': 2, 'active_fn': 'relu', 'alpha': 1.0, 'beta': 0.1, 'gamma': 0, 'out_path': 'mortgage/results/'}
    mortgage/results//itr_0 (a:1.0 b:0.1 c:0)
    MAIN TRAINING ...
    EVALUATION TIMES: [3, 7, 17]
    || ITR: 1000 | Loss: [1m[33m2.0210[0m
    updated.... average c-index = 0.8360
    || ITR: 2000 | Loss: [1m[33m1.7633[0m
    updated.... average c-index = 0.8537
    || ITR: 3000 | Loss: [1m[33m1.6769[0m
    updated.... average c-index = 0.8645
    || ITR: 4000 | Loss: [1m[33m1.6362[0m
    updated.... average c-index = 0.8691
    || ITR: 5000 | Loss: [1m[33m1.5891[0m
    updated.... average c-index = 0.8763
    || ITR: 6000 | Loss: [1m[33m1.5494[0m
    updated.... average c-index = 0.8807
    || ITR: 7000 | Loss: [1m[33m1.5240[0m
    updated.... average c-index = 0.8845
    || ITR: 8000 | Loss: [1m[33m1.5193[0m
    updated.... average c-index = 0.8866
    || ITR: 9000 | Loss: [1m[33m1.4988[0m
    updated.... average c-index = 0.8911
    || ITR: 10000 | Loss: [1m[33m1.4685[0m
    updated.... average c-index = 0.8936
    || ITR: 11000 | Loss: [1m[33m1.4481[0m
    || ITR: 12000 | Loss: [1m[33m1.4413[0m
    || ITR: 13000 | Loss: [1m[33m1.4270[0m
    updated.... average c-index = 0.8966
    || ITR: 14000 | Loss: [1m[33m1.4213[0m
    updated.... average c-index = 0.8976
    || ITR: 15000 | Loss: [1m[33m1.3964[0m
    || ITR: 16000 | Loss: [1m[33m1.3897[0m
    updated.... average c-index = 0.8999
    || ITR: 17000 | Loss: [1m[33m1.3597[0m
    updated.... average c-index = 0.9001
    || ITR: 18000 | Loss: [1m[33m1.3572[0m
    updated.... average c-index = 0.9004
    || ITR: 19000 | Loss: [1m[33m1.3459[0m
    updated.... average c-index = 0.9019
    || ITR: 20000 | Loss: [1m[33m1.3750[0m
    || ITR: 21000 | Loss: [1m[33m1.3273[0m
    updated.... average c-index = 0.9036
    || ITR: 22000 | Loss: [1m[33m1.3443[0m
    || ITR: 23000 | Loss: [1m[33m1.3347[0m
    || ITR: 24000 | Loss: [1m[33m1.3313[0m
    updated.... average c-index = 0.9039
    || ITR: 25000 | Loss: [1m[33m1.3131[0m
    || ITR: 26000 | Loss: [1m[33m1.3082[0m
    updated.... average c-index = 0.9041
    || ITR: 27000 | Loss: [1m[33m1.3118[0m
    || ITR: 28000 | Loss: [1m[33m1.3028[0m
    updated.... average c-index = 0.9055
    || ITR: 29000 | Loss: [1m[33m1.3035[0m
    || ITR: 30000 | Loss: [1m[33m1.2787[0m
    || ITR: 31000 | Loss: [1m[33m1.2914[0m
    || ITR: 32000 | Loss: [1m[33m1.2957[0m
    updated.... average c-index = 0.9061
    || ITR: 33000 | Loss: [1m[33m1.2822[0m
    || ITR: 34000 | Loss: [1m[33m1.2822[0m
    updated.... average c-index = 0.9069
    || ITR: 35000 | Loss: [1m[33m1.2678[0m
    || ITR: 36000 | Loss: [1m[33m1.2836[0m
    || ITR: 37000 | Loss: [1m[33m1.2749[0m
    || ITR: 38000 | Loss: [1m[33m1.2703[0m
    || ITR: 39000 | Loss: [1m[33m1.2607[0m
    updated.... average c-index = 0.9079
    || ITR: 40000 | Loss: [1m[33m1.2553[0m
    || ITR: 41000 | Loss: [1m[33m1.2688[0m
    || ITR: 42000 | Loss: [1m[33m1.2542[0m
    || ITR: 43000 | Loss: [1m[33m1.2580[0m
    updated.... average c-index = 0.9083
    || ITR: 44000 | Loss: [1m[33m1.2493[0m
    updated.... average c-index = 0.9083
    || ITR: 45000 | Loss: [1m[33m1.2423[0m
    updated.... average c-index = 0.9088
    || ITR: 46000 | Loss: [1m[33m1.2262[0m
    || ITR: 47000 | Loss: [1m[33m1.2448[0m
    || ITR: 48000 | Loss: [1m[33m1.2341[0m
    updated.... average c-index = 0.9092
    || ITR: 49000 | Loss: [1m[33m1.2391[0m
    || ITR: 50000 | Loss: [1m[33m1.2371[0m
    Current best: 0.9239002196511809
    OUTER_ITERATION: 0
    Random search... itr: 4
    {'mb_size': 64, 'iteration': 50000, 'keep_prob': 0.6, 'lr_train': 0.0001, 'h_dim_shared': 50, 'h_dim_CS': 50, 'num_layers_shared': 1, 'num_layers_CS': 2, 'active_fn': 'relu', 'alpha': 1.0, 'beta': 5.0, 'gamma': 0, 'out_path': 'mortgage/results/'}
    mortgage/results//itr_0 (a:1.0 b:5.0 c:0)
    MAIN TRAINING ...
    EVALUATION TIMES: [3, 7, 17]
    || ITR: 1000 | Loss: [1m[33m40.6204[0m
    updated.... average c-index = 0.8246
    || ITR: 2000 | Loss: [1m[33m32.8358[0m
    updated.... average c-index = 0.8489
    || ITR: 3000 | Loss: [1m[33m28.8008[0m
    updated.... average c-index = 0.8559
    || ITR: 4000 | Loss: [1m[33m27.8822[0m
    updated.... average c-index = 0.8621
    || ITR: 5000 | Loss: [1m[33m25.3325[0m
    updated.... average c-index = 0.8669
    || ITR: 6000 | Loss: [1m[33m24.8330[0m
    updated.... average c-index = 0.8719
    || ITR: 7000 | Loss: [1m[33m23.5804[0m
    updated.... average c-index = 0.8741
    || ITR: 8000 | Loss: [1m[33m22.6519[0m
    updated.... average c-index = 0.8765
    || ITR: 9000 | Loss: [1m[33m21.8727[0m
    updated.... average c-index = 0.8819
    || ITR: 10000 | Loss: [1m[33m21.5534[0m
    updated.... average c-index = 0.8841
    || ITR: 11000 | Loss: [1m[33m21.3036[0m
    updated.... average c-index = 0.8877
    || ITR: 12000 | Loss: [1m[33m20.8615[0m
    updated.... average c-index = 0.8886
    || ITR: 13000 | Loss: [1m[33m20.3721[0m
    updated.... average c-index = 0.8898
    || ITR: 14000 | Loss: [1m[33m19.2156[0m
    updated.... average c-index = 0.8910
    || ITR: 15000 | Loss: [1m[33m19.8268[0m
    updated.... average c-index = 0.8943
    || ITR: 16000 | Loss: [1m[33m19.1501[0m
    || ITR: 17000 | Loss: [1m[33m18.9285[0m
    || ITR: 18000 | Loss: [1m[33m18.4639[0m
    || ITR: 19000 | Loss: [1m[33m18.2587[0m
    || ITR: 20000 | Loss: [1m[33m18.1981[0m
    || ITR: 21000 | Loss: [1m[33m17.8358[0m
    Current best: 0.9239002196511809
    OUTER_ITERATION: 1
    Random search... itr: 0
    {'mb_size': 128, 'iteration': 50000, 'keep_prob': 0.6, 'lr_train': 0.0001, 'h_dim_shared': 300, 'h_dim_CS': 300, 'num_layers_shared': 3, 'num_layers_CS': 1, 'active_fn': 'relu', 'alpha': 1.0, 'beta': 3.0, 'gamma': 0, 'out_path': 'mortgage/results/'}
    mortgage/results//itr_1 (a:1.0 b:3.0 c:0)
    MAIN TRAINING ...
    EVALUATION TIMES: [3, 7, 17]
    || ITR: 1000 | Loss: [1m[33m32.6632[0m
    updated.... average c-index = 0.8491
    || ITR: 2000 | Loss: [1m[33m25.6581[0m
    updated.... average c-index = 0.8666
    || ITR: 3000 | Loss: [1m[33m23.3444[0m
    updated.... average c-index = 0.8776
    || ITR: 4000 | Loss: [1m[33m21.6084[0m
    updated.... average c-index = 0.8822
    || ITR: 5000 | Loss: [1m[33m20.4474[0m
    updated.... average c-index = 0.8852
    || ITR: 6000 | Loss: [1m[33m19.6392[0m
    updated.... average c-index = 0.8907
    || ITR: 7000 | Loss: [1m[33m18.8850[0m
    updated.... average c-index = 0.8926
    || ITR: 8000 | Loss: [1m[33m18.0847[0m
    updated.... average c-index = 0.8927
    || ITR: 9000 | Loss: [1m[33m17.6617[0m
    || ITR: 10000 | Loss: [1m[33m17.2383[0m
    updated.... average c-index = 0.8952
    || ITR: 11000 | Loss: [1m[33m16.6899[0m
    || ITR: 12000 | Loss: [1m[33m16.6172[0m
    || ITR: 13000 | Loss: [1m[33m15.9579[0m
    || ITR: 14000 | Loss: [1m[33m15.3831[0m
    || ITR: 15000 | Loss: [1m[33m15.2822[0m
    || ITR: 16000 | Loss: [1m[33m15.0214[0m
    Current best: 0.895198148911295
    OUTER_ITERATION: 1
    Random search... itr: 1
    {'mb_size': 128, 'iteration': 50000, 'keep_prob': 0.6, 'lr_train': 0.0001, 'h_dim_shared': 100, 'h_dim_CS': 100, 'num_layers_shared': 3, 'num_layers_CS': 2, 'active_fn': 'relu', 'alpha': 1.0, 'beta': 0.5, 'gamma': 0, 'out_path': 'mortgage/results/'}
    mortgage/results//itr_1 (a:1.0 b:0.5 c:0)
    MAIN TRAINING ...
    EVALUATION TIMES: [3, 7, 17]
    || ITR: 1000 | Loss: [1m[33m8.5076[0m
    updated.... average c-index = 0.8294
    || ITR: 2000 | Loss: [1m[33m6.7903[0m
    updated.... average c-index = 0.8433
    || ITR: 3000 | Loss: [1m[33m6.1928[0m
    updated.... average c-index = 0.8518
    || ITR: 4000 | Loss: [1m[33m5.8659[0m
    updated.... average c-index = 0.8595
    || ITR: 5000 | Loss: [1m[33m5.4855[0m
    updated.... average c-index = 0.8659
    || ITR: 6000 | Loss: [1m[33m5.3695[0m
    updated.... average c-index = 0.8676
    || ITR: 7000 | Loss: [1m[33m5.2016[0m
    updated.... average c-index = 0.8754
    || ITR: 8000 | Loss: [1m[33m5.0098[0m
    updated.... average c-index = 0.8785
    || ITR: 9000 | Loss: [1m[33m4.8610[0m
    updated.... average c-index = 0.8820
    || ITR: 10000 | Loss: [1m[33m4.7613[0m
    || ITR: 11000 | Loss: [1m[33m4.6220[0m
    updated.... average c-index = 0.8863
    || ITR: 12000 | Loss: [1m[33m4.5974[0m
    updated.... average c-index = 0.8873
    || ITR: 13000 | Loss: [1m[33m4.5248[0m
    updated.... average c-index = 0.8875
    || ITR: 14000 | Loss: [1m[33m4.4395[0m
    updated.... average c-index = 0.8885
    || ITR: 15000 | Loss: [1m[33m4.3565[0m
    updated.... average c-index = 0.8894
    || ITR: 16000 | Loss: [1m[33m4.3201[0m
    updated.... average c-index = 0.8898
    || ITR: 17000 | Loss: [1m[33m4.3331[0m
    || ITR: 18000 | Loss: [1m[33m4.2138[0m
    || ITR: 19000 | Loss: [1m[33m4.2063[0m
    updated.... average c-index = 0.8911
    || ITR: 20000 | Loss: [1m[33m4.1569[0m
    || ITR: 21000 | Loss: [1m[33m4.1029[0m
    || ITR: 22000 | Loss: [1m[33m4.1126[0m
    updated.... average c-index = 0.8914
    || ITR: 23000 | Loss: [1m[33m4.0406[0m
    || ITR: 24000 | Loss: [1m[33m4.0112[0m
    || ITR: 25000 | Loss: [1m[33m4.0339[0m
    || ITR: 26000 | Loss: [1m[33m4.0366[0m
    || ITR: 27000 | Loss: [1m[33m3.9478[0m
    || ITR: 28000 | Loss: [1m[33m3.9702[0m
    Current best: 0.895198148911295
    OUTER_ITERATION: 1
    Random search... itr: 2
    {'mb_size': 64, 'iteration': 50000, 'keep_prob': 0.6, 'lr_train': 0.0001, 'h_dim_shared': 200, 'h_dim_CS': 300, 'num_layers_shared': 2, 'num_layers_CS': 1, 'active_fn': 'relu', 'alpha': 1.0, 'beta': 5.0, 'gamma': 0, 'out_path': 'mortgage/results/'}
    mortgage/results//itr_1 (a:1.0 b:5.0 c:0)
    MAIN TRAINING ...
    EVALUATION TIMES: [3, 7, 17]
    || ITR: 1000 | Loss: [1m[33m28.9439[0m
    updated.... average c-index = 0.8407
    || ITR: 2000 | Loss: [1m[33m23.2324[0m
    updated.... average c-index = 0.8579
    || ITR: 3000 | Loss: [1m[33m21.0208[0m
    updated.... average c-index = 0.8661
    || ITR: 4000 | Loss: [1m[33m19.7210[0m
    updated.... average c-index = 0.8698
    || ITR: 5000 | Loss: [1m[33m18.7016[0m
    updated.... average c-index = 0.8747
    || ITR: 6000 | Loss: [1m[33m18.3012[0m
    updated.... average c-index = 0.8749
    || ITR: 7000 | Loss: [1m[33m17.7508[0m
    updated.... average c-index = 0.8790
    || ITR: 8000 | Loss: [1m[33m17.0634[0m
    updated.... average c-index = 0.8850
    || ITR: 9000 | Loss: [1m[33m16.9440[0m
    updated.... average c-index = 0.8882
    || ITR: 10000 | Loss: [1m[33m16.2292[0m
    || ITR: 11000 | Loss: [1m[33m16.0796[0m
    updated.... average c-index = 0.8895
    || ITR: 12000 | Loss: [1m[33m15.6636[0m
    updated.... average c-index = 0.8915
    || ITR: 13000 | Loss: [1m[33m15.3829[0m
    updated.... average c-index = 0.8948
    || ITR: 14000 | Loss: [1m[33m15.2636[0m
    || ITR: 15000 | Loss: [1m[33m14.8885[0m
    updated.... average c-index = 0.8951
    || ITR: 16000 | Loss: [1m[33m14.7829[0m
    updated.... average c-index = 0.8967
    || ITR: 17000 | Loss: [1m[33m14.6141[0m
    || ITR: 18000 | Loss: [1m[33m14.3207[0m
    updated.... average c-index = 0.8981
    || ITR: 19000 | Loss: [1m[33m14.1675[0m
    || ITR: 20000 | Loss: [1m[33m13.9592[0m
    || ITR: 21000 | Loss: [1m[33m13.6649[0m
    updated.... average c-index = 0.8992
    || ITR: 22000 | Loss: [1m[33m13.5199[0m
    updated.... average c-index = 0.8999
    || ITR: 23000 | Loss: [1m[33m13.1917[0m
    || ITR: 24000 | Loss: [1m[33m13.3917[0m
    || ITR: 25000 | Loss: [1m[33m12.9680[0m
    || ITR: 26000 | Loss: [1m[33m13.1010[0m
    || ITR: 27000 | Loss: [1m[33m12.7412[0m
    || ITR: 28000 | Loss: [1m[33m12.5700[0m
    Current best: 0.8999294939096446
    OUTER_ITERATION: 1
    Random search... itr: 3
    {'mb_size': 64, 'iteration': 50000, 'keep_prob': 0.6, 'lr_train': 0.0001, 'h_dim_shared': 300, 'h_dim_CS': 200, 'num_layers_shared': 3, 'num_layers_CS': 1, 'active_fn': 'relu', 'alpha': 1.0, 'beta': 3.0, 'gamma': 0, 'out_path': 'mortgage/results/'}
    mortgage/results//itr_1 (a:1.0 b:3.0 c:0)
    MAIN TRAINING ...
    EVALUATION TIMES: [3, 7, 17]
    || ITR: 1000 | Loss: [1m[33m19.0210[0m
    updated.... average c-index = 0.8518
    || ITR: 2000 | Loss: [1m[33m14.7452[0m
    updated.... average c-index = 0.8588
    || ITR: 3000 | Loss: [1m[33m13.4368[0m
    updated.... average c-index = 0.8679
    || ITR: 4000 | Loss: [1m[33m12.6216[0m
    updated.... average c-index = 0.8790
    || ITR: 5000 | Loss: [1m[33m11.9418[0m
    updated.... average c-index = 0.8834
    || ITR: 6000 | Loss: [1m[33m11.6273[0m
    updated.... average c-index = 0.8849
    || ITR: 7000 | Loss: [1m[33m11.1290[0m
    updated.... average c-index = 0.8879
    || ITR: 8000 | Loss: [1m[33m10.6850[0m
    updated.... average c-index = 0.8901
    || ITR: 9000 | Loss: [1m[33m10.8297[0m
    updated.... average c-index = 0.8902
    || ITR: 10000 | Loss: [1m[33m10.2206[0m
    updated.... average c-index = 0.8921
    || ITR: 11000 | Loss: [1m[33m10.0748[0m
    updated.... average c-index = 0.8922
    || ITR: 12000 | Loss: [1m[33m10.0051[0m
    updated.... average c-index = 0.8935
    || ITR: 13000 | Loss: [1m[33m9.8905[0m
    updated.... average c-index = 0.8938
    || ITR: 14000 | Loss: [1m[33m9.8016[0m
    updated.... average c-index = 0.8955
    || ITR: 15000 | Loss: [1m[33m9.4203[0m
    || ITR: 16000 | Loss: [1m[33m9.3422[0m
    || ITR: 17000 | Loss: [1m[33m9.1040[0m
    || ITR: 18000 | Loss: [1m[33m8.9606[0m
    || ITR: 19000 | Loss: [1m[33m8.7415[0m
    updated.... average c-index = 0.8961
    || ITR: 20000 | Loss: [1m[33m8.7848[0m
    || ITR: 21000 | Loss: [1m[33m8.5457[0m
    || ITR: 22000 | Loss: [1m[33m8.5382[0m
    || ITR: 23000 | Loss: [1m[33m8.4145[0m
    || ITR: 24000 | Loss: [1m[33m8.4060[0m
    || ITR: 25000 | Loss: [1m[33m8.2287[0m
    Current best: 0.8999294939096446
    OUTER_ITERATION: 1
    Random search... itr: 4
    {'mb_size': 32, 'iteration': 50000, 'keep_prob': 0.6, 'lr_train': 0.0001, 'h_dim_shared': 50, 'h_dim_CS': 100, 'num_layers_shared': 2, 'num_layers_CS': 5, 'active_fn': 'relu', 'alpha': 1.0, 'beta': 0.1, 'gamma': 0, 'out_path': 'mortgage/results/'}
    mortgage/results//itr_1 (a:1.0 b:0.1 c:0)
    MAIN TRAINING ...
    EVALUATION TIMES: [3, 7, 17]
    || ITR: 1000 | Loss: [1m[33m1.8761[0m
    updated.... average c-index = 0.7185
    || ITR: 2000 | Loss: [1m[33m1.7761[0m
    updated.... average c-index = 0.7951
    || ITR: 3000 | Loss: [1m[33m1.7376[0m
    updated.... average c-index = 0.8377
    || ITR: 4000 | Loss: [1m[33m1.6776[0m
    updated.... average c-index = 0.8519
    || ITR: 5000 | Loss: [1m[33m1.6078[0m
    updated.... average c-index = 0.8597
    || ITR: 6000 | Loss: [1m[33m1.5684[0m
    updated.... average c-index = 0.8680
    || ITR: 7000 | Loss: [1m[33m1.5266[0m
    updated.... average c-index = 0.8698
    || ITR: 8000 | Loss: [1m[33m1.5258[0m
    updated.... average c-index = 0.8707
    || ITR: 9000 | Loss: [1m[33m1.5017[0m
    updated.... average c-index = 0.8723
    || ITR: 10000 | Loss: [1m[33m1.4941[0m
    updated.... average c-index = 0.8769
    || ITR: 11000 | Loss: [1m[33m1.4919[0m
    updated.... average c-index = 0.8773
    || ITR: 12000 | Loss: [1m[33m1.4509[0m
    updated.... average c-index = 0.8799
    || ITR: 13000 | Loss: [1m[33m1.4784[0m
    updated.... average c-index = 0.8819
    || ITR: 14000 | Loss: [1m[33m1.4607[0m
    updated.... average c-index = 0.8825
    || ITR: 15000 | Loss: [1m[33m1.4089[0m
    updated.... average c-index = 0.8846
    || ITR: 16000 | Loss: [1m[33m1.4349[0m
    updated.... average c-index = 0.8855
    || ITR: 17000 | Loss: [1m[33m1.4265[0m
    updated.... average c-index = 0.8861
    || ITR: 18000 | Loss: [1m[33m1.4024[0m
    updated.... average c-index = 0.8871
    || ITR: 19000 | Loss: [1m[33m1.4015[0m
    || ITR: 20000 | Loss: [1m[33m1.4057[0m
    updated.... average c-index = 0.8893
    || ITR: 21000 | Loss: [1m[33m1.3930[0m
    updated.... average c-index = 0.8919
    || ITR: 22000 | Loss: [1m[33m1.3965[0m
    || ITR: 23000 | Loss: [1m[33m1.3836[0m
    updated.... average c-index = 0.8920
    || ITR: 24000 | Loss: [1m[33m1.3765[0m
    updated.... average c-index = 0.8942
    || ITR: 25000 | Loss: [1m[33m1.3833[0m
    || ITR: 26000 | Loss: [1m[33m1.3709[0m
    updated.... average c-index = 0.8956
    || ITR: 27000 | Loss: [1m[33m1.3578[0m
    updated.... average c-index = 0.8971
    || ITR: 28000 | Loss: [1m[33m1.3559[0m
    || ITR: 29000 | Loss: [1m[33m1.3584[0m
    updated.... average c-index = 0.8988
    || ITR: 30000 | Loss: [1m[33m1.3456[0m
    || ITR: 31000 | Loss: [1m[33m1.3331[0m
    || ITR: 32000 | Loss: [1m[33m1.3552[0m
    updated.... average c-index = 0.8996
    || ITR: 33000 | Loss: [1m[33m1.3306[0m
    || ITR: 34000 | Loss: [1m[33m1.3133[0m
    || ITR: 35000 | Loss: [1m[33m1.3201[0m
    || ITR: 36000 | Loss: [1m[33m1.3181[0m
    updated.... average c-index = 0.9017
    || ITR: 37000 | Loss: [1m[33m1.3034[0m
    || ITR: 38000 | Loss: [1m[33m1.3244[0m
    || ITR: 39000 | Loss: [1m[33m1.2881[0m
    || ITR: 40000 | Loss: [1m[33m1.2973[0m
    || ITR: 41000 | Loss: [1m[33m1.2950[0m
    || ITR: 42000 | Loss: [1m[33m1.3001[0m
    updated.... average c-index = 0.9037
    || ITR: 43000 | Loss: [1m[33m1.2790[0m
    || ITR: 44000 | Loss: [1m[33m1.3008[0m
    || ITR: 45000 | Loss: [1m[33m1.2949[0m
    || ITR: 46000 | Loss: [1m[33m1.2801[0m
    || ITR: 47000 | Loss: [1m[33m1.2683[0m
    || ITR: 48000 | Loss: [1m[33m1.2904[0m
    Current best: 0.903706588990325
    


```python
for itr in range(OUT_ITERATION):
    
    if not os.path.exists(out_path + '/itr_' + str(itr) + '/'):
        os.makedirs(out_path + '/itr_' + str(itr) + '/')

    max_valid = 0.
    log_name = out_path + '/itr_' + str(itr) + '/hyperparameters_log.txt'

    for r_itr in range(RS_ITERATION):
        print('OUTER_ITERATION: ' + str(itr))
        print('Random search... itr: ' + str(r_itr))
        new_parser = get_random_hyperparameters(out_path)
        print(new_parser)

        # get validation performance given the hyperparameters
        tmp_max = get_main.get_valid_performance(DATA, MASK, new_parser, itr, eval_times, MAX_VALUE=max_valid)

        if tmp_max > max_valid:
            max_valid = tmp_max
            max_parser = new_parser
            save_logging(max_parser, log_name)  #save the hyperparameters if this provides the maximum validation performance

        print('Current best: ' + str(max_valid))
```

    OUTER_ITERATION: 0
    Random search... itr: 0
    {'mb_size': 64, 'iteration': 50000, 'keep_prob': 0.6, 'lr_train': 0.0001, 'h_dim_shared': 300, 'h_dim_CS': 300, 'num_layers_shared': 1, 'num_layers_CS': 3, 'active_fn': 'relu', 'alpha': 1.0, 'beta': 3.0, 'gamma': 0, 'out_path': 'mortgage/results_cr/'}
    mortgage/results_cr//itr_0 (a:1.0 b:3.0 c:0)
    MAIN TRAINING ...
    EVALUATION TIMES: [3, 7, 17]
    || ITR: 1000 | Loss: [1m[33m31.2018[0m
    updated.... average c-index = 0.8560
    || ITR: 2000 | Loss: [1m[33m23.2983[0m
    updated.... average c-index = 0.8804
    || ITR: 3000 | Loss: [1m[33m20.6790[0m
    updated.... average c-index = 0.8869
    || ITR: 4000 | Loss: [1m[33m18.9983[0m
    || ITR: 5000 | Loss: [1m[33m18.0706[0m
    updated.... average c-index = 0.8908
    || ITR: 6000 | Loss: [1m[33m17.3130[0m
    updated.... average c-index = 0.8922
    || ITR: 7000 | Loss: [1m[33m16.6637[0m
    updated.... average c-index = 0.8936
    || ITR: 8000 | Loss: [1m[33m16.0250[0m
    || ITR: 9000 | Loss: [1m[33m15.6862[0m
    || ITR: 10000 | Loss: [1m[33m15.2763[0m
    || ITR: 11000 | Loss: [1m[33m14.9472[0m
    || ITR: 12000 | Loss: [1m[33m14.6367[0m
    || ITR: 13000 | Loss: [1m[33m14.5165[0m
    Current best: 0.8936147638202074
    OUTER_ITERATION: 0
    Random search... itr: 1
    {'mb_size': 64, 'iteration': 50000, 'keep_prob': 0.6, 'lr_train': 0.0001, 'h_dim_shared': 200, 'h_dim_CS': 100, 'num_layers_shared': 2, 'num_layers_CS': 2, 'active_fn': 'relu', 'alpha': 1.0, 'beta': 1.0, 'gamma': 0, 'out_path': 'mortgage/results_cr/'}
    mortgage/results_cr//itr_0 (a:1.0 b:1.0 c:0)
    MAIN TRAINING ...
    EVALUATION TIMES: [3, 7, 17]
    || ITR: 1000 | Loss: [1m[33m15.1956[0m
    updated.... average c-index = 0.8237
    || ITR: 2000 | Loss: [1m[33m12.5315[0m
    updated.... average c-index = 0.8543
    || ITR: 3000 | Loss: [1m[33m11.1691[0m
    updated.... average c-index = 0.8711
    || ITR: 4000 | Loss: [1m[33m10.3862[0m
    updated.... average c-index = 0.8814
    || ITR: 5000 | Loss: [1m[33m9.8140[0m
    updated.... average c-index = 0.8873
    || ITR: 6000 | Loss: [1m[33m9.3900[0m
    updated.... average c-index = 0.8924
    || ITR: 7000 | Loss: [1m[33m9.0458[0m
    || ITR: 8000 | Loss: [1m[33m8.7308[0m
    || ITR: 9000 | Loss: [1m[33m8.5158[0m
    || ITR: 10000 | Loss: [1m[33m8.3060[0m
    || ITR: 11000 | Loss: [1m[33m8.1498[0m
    || ITR: 12000 | Loss: [1m[33m8.0461[0m
    Current best: 0.8936147638202074
    


```python
eval_time = [int(np.percentile(tr_time, 25)), int(np.percentile(tr_time, 50)), int(np.percentile(tr_time, 75))]
```


```python
print( "MAIN TRAINING ...")
print( "EVALUATION TIMES: " + str(eval_time))
```

    MAIN TRAINING ...
    EVALUATION TIMES: [3, 7, 17]
    


```python
##### DATA & MASK
(data, time, label)  = DATA
(mask1, mask2)       = MASK

x_dim                       = np.shape(data)[1]
_, num_Event, num_Category  = np.shape(mask1)  # dim of mask1: [subj, Num_Event, Num_Category]
    
ACTIVATION_FN               = {'relu': tf.nn.relu, 'elu': tf.nn.elu, 'tanh': tf.nn.tanh}

##### HYPER-PARAMETERS
new_parser = get_random_hyperparameters(out_path)
print(new_parser)
mb_size                     = new_parser['mb_size']

iteration                   = new_parser['iteration']

keep_prob                   = new_parser['keep_prob']
lr_train                    = new_parser['lr_train']


alpha                       = new_parser['alpha']  #for log-likelihood loss
beta                        = new_parser['beta']  #for ranking loss
gamma                       = new_parser['gamma']  #for RNN-prediction loss
parameter_name              = 'a' + str('%02.0f' %(10*alpha)) + 'b' + str('%02.0f' %(10*beta)) + 'c' + str('%02.0f' %(10*gamma))

initial_W                   = tf.contrib.layers.xavier_initializer()


##### MAKE DICTIONARIES
# INPUT DIMENSIONS
input_dims                  = { 'x_dim'         : x_dim,
                                'num_Event'     : num_Event,
                                'num_Category'  : num_Category}

# NETWORK HYPER-PARMETERS
network_settings            = { 'h_dim_shared'       : new_parser['h_dim_shared'],
                                'num_layers_shared'  : new_parser['num_layers_shared'],
                                'h_dim_CS'           : new_parser['h_dim_CS'],
                                'num_layers_CS'      : new_parser['num_layers_CS'],
                                'active_fn'          : ACTIVATION_FN[new_parser['active_fn']],
                                'initial_W'          : initial_W }


```

    {'mb_size': 128, 'iteration': 50000, 'keep_prob': 0.6, 'lr_train': 0.0001, 'h_dim_shared': 200, 'h_dim_CS': 300, 'num_layers_shared': 2, 'num_layers_CS': 1, 'active_fn': 'relu', 'alpha': 1.0, 'beta': 1.0, 'gamma': 0, 'out_path': 'mortgage/results_cr/'}
    


```python
##### CREATE DEEPHIT NETWORK
tf.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

model = Model_DeepHit(sess, "DeepHit", input_dims, network_settings)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(out_path, sess.graph)
```


```python
### TRAINING-TESTING SPLIT
(tr_data,te_data, tr_time,te_time, tr_label,te_label, 
    tr_mask1,te_mask1, tr_mask2,te_mask2)  = train_test_split(dhdata_cr, time, label, mask1, mask2, test_size=0.20, random_state=seed) 
    

(tr_data,va_data, tr_time,va_time, tr_label,va_label, 
    tr_mask1,va_mask1, tr_mask2,va_mask2)  = train_test_split(tr_data, tr_time, tr_label, tr_mask1, tr_mask2, test_size=0.20, random_state=seed) 
```


```python
### TRAINING - MAIN
print( "MAIN TRAINING ...")
print( "EVALUATION TIMES: " + str(eval_time))

iteration = 5
avg_loss = 0
max_valid = -99
stop_flag = 0
for itr in range(iteration):
    if stop_flag > 5: #for faster early stopping
        break
    else:
        x_mb, k_mb, t_mb, m1_mb, m2_mb = f_get_minibatch(mb_size, tr_data, tr_label, tr_time, tr_mask1, tr_mask2)
        DATA = (x_mb, k_mb, t_mb)
        MASK = (m1_mb, m2_mb)
        PARAMETERS = (alpha, beta, gamma)
        _, loss_curr = model.train(DATA, MASK, PARAMETERS, keep_prob, lr_train)
        avg_loss += loss_curr/1000
                
        if (itr+1)%1000 == 0:
            print('|| ITR: ' + str('%04d' % (itr + 1)) + ' | Loss: ' + colored(str('%.4f' %(avg_loss)), 'yellow' , attrs=['bold']))
            avg_loss = 0

        ### VALIDATION  (based on average C-index of our interest)
        if (itr+1)%1000 == 0:
        ### PREDICTION
            pred = model.predict(va_data)

            ### EVALUATION
            va_result1 = np.zeros([num_Event, len(eval_time)])

            for t, t_time in enumerate(eval_time):
                eval_horizon = int(t_time)

                if eval_horizon >= num_Category:
                    print('ERROR: evaluation horizon is out of range')
                    va_result1[:, t] = va_result2[:, t] = -1
                else:
                    risk = np.sum(pred[:,:,:(eval_horizon+1)], axis=2) #risk score until eval_time
                    for k in range(num_Event):
                        # va_result1[k, t] = c_index(risk[:,k], va_time, (va_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
                        va_result1[k, t] = weighted_c_index(tr_time, (tr_label[:,0] == k+1).astype(int), risk[:,k], va_time, (va_label[:,0] == k+1).astype(int), eval_horizon)
            tmp_valid = np.mean(va_result1)


            if tmp_valid >  max_valid:
                stop_flag = 0
                max_valid = tmp_valid
                print( 'updated.... average c-index = ' + str('%.4f' %(tmp_valid)))

                if max_valid > MAX_VALUE:
                    saver.save(sess, file_path_final + '/models/model_itr_' + str(out_itr))
            else:
                stop_flag += 1
```

    MAIN TRAINING ...
    EVALUATION TIMES: [3, 7, 17]
    


```python
avg_loss = 0
max_valid = -99
stop_flag = 0
for itr in range(iteration):
        if stop_flag > 5: #for faster early stopping
            break
        else:
            x_mb, k_mb, t_mb, m1_mb, m2_mb = f_get_minibatch(mb_size, tr_data, tr_label, tr_time, tr_mask1, tr_mask2)
            DATA = (x_mb, k_mb, t_mb)
            MASK = (m1_mb, m2_mb)
            PARAMETERS = (alpha, beta, gamma)
            _, loss_curr = model.train(DATA, MASK, PARAMETERS, keep_prob, lr_train)
            avg_loss += loss_curr/1000
                
            if (itr+1)%1000 == 0:
                print('|| ITR: ' + str('%04d' % (itr + 1)) + ' | Loss: ' + colored(str('%.4f' %(avg_loss)), 'yellow' , attrs=['bold']))
                avg_loss = 0

            ### VALIDATION  (based on average C-index of our interest)
            if (itr+1)%1000 == 0:
                ### PREDICTION
                pred = model.predict(va_data)

                ### EVALUATION
                va_result1 = np.zeros([num_Event, len(eval_time)])

                for t, t_time in enumerate(eval_time):
                    eval_horizon = int(t_time)

                    if eval_horizon >= num_Category:
                        print('ERROR: evaluation horizon is out of range')
                        va_result1[:, t] = va_result2[:, t] = -1
                    else:
                        risk = np.sum(pred[:,:,:(eval_horizon+1)], axis=2) #risk score until eval_time
                        for k in range(num_Event):
                            # va_result1[k, t] = c_index(risk[:,k], va_time, (va_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
                            va_result1[k, t] = weighted_c_index(tr_time, (tr_label[:,0] == k+1).astype(int), risk[:,k], va_time, (va_label[:,0] == k+1).astype(int), eval_horizon)
                tmp_valid = np.mean(va_result1)


                if tmp_valid >  max_valid:
                    stop_flag = 0
                    max_valid = tmp_valid
                    print( 'updated.... average c-index = ' + str('%.4f' %(tmp_valid)))
                else:
                    stop_flag += 1       
```

    || ITR: 1000 | Loss: [1m[33m3.6786[0m
    updated.... average c-index = 0.5765
    || ITR: 2000 | Loss: [1m[33m3.6032[0m
    updated.... average c-index = 0.6830
    || ITR: 3000 | Loss: [1m[33m3.5376[0m
    updated.... average c-index = 0.7644
    || ITR: 4000 | Loss: [1m[33m3.4251[0m
    updated.... average c-index = 0.8191
    || ITR: 5000 | Loss: [1m[33m3.3710[0m
    updated.... average c-index = 0.8454
    || ITR: 6000 | Loss: [1m[33m3.2881[0m
    updated.... average c-index = 0.8565
    || ITR: 7000 | Loss: [1m[33m3.2941[0m
    updated.... average c-index = 0.8651
    || ITR: 8000 | Loss: [1m[33m3.2089[0m
    updated.... average c-index = 0.8700
    || ITR: 9000 | Loss: [1m[33m3.1626[0m
    updated.... average c-index = 0.8729
    || ITR: 10000 | Loss: [1m[33m3.1188[0m
    updated.... average c-index = 0.8756
    || ITR: 11000 | Loss: [1m[33m3.1050[0m
    updated.... average c-index = 0.8772
    || ITR: 12000 | Loss: [1m[33m3.1185[0m
    updated.... average c-index = 0.8779
    || ITR: 13000 | Loss: [1m[33m3.1046[0m
    updated.... average c-index = 0.8792
    || ITR: 14000 | Loss: [1m[33m3.0387[0m
    updated.... average c-index = 0.8803
    || ITR: 15000 | Loss: [1m[33m3.0431[0m
    updated.... average c-index = 0.8809
    || ITR: 16000 | Loss: [1m[33m3.0408[0m
    updated.... average c-index = 0.8816
    || ITR: 17000 | Loss: [1m[33m3.0407[0m
    updated.... average c-index = 0.8816
    || ITR: 18000 | Loss: [1m[33m2.9724[0m
    || ITR: 19000 | Loss: [1m[33m2.9378[0m
    updated.... average c-index = 0.8820
    || ITR: 20000 | Loss: [1m[33m2.9608[0m
    updated.... average c-index = 0.8830
    || ITR: 21000 | Loss: [1m[33m3.0036[0m
    || ITR: 22000 | Loss: [1m[33m2.9624[0m
    || ITR: 23000 | Loss: [1m[33m2.9286[0m
    || ITR: 24000 | Loss: [1m[33m2.9021[0m
    updated.... average c-index = 0.8831
    || ITR: 25000 | Loss: [1m[33m2.9136[0m
    updated.... average c-index = 0.8832
    || ITR: 26000 | Loss: [1m[33m2.8937[0m
    || ITR: 27000 | Loss: [1m[33m2.8854[0m
    updated.... average c-index = 0.8834
    || ITR: 28000 | Loss: [1m[33m2.8483[0m
    updated.... average c-index = 0.8842
    || ITR: 29000 | Loss: [1m[33m2.8295[0m
    updated.... average c-index = 0.8853
    || ITR: 30000 | Loss: [1m[33m2.9004[0m
    updated.... average c-index = 0.8857
    || ITR: 31000 | Loss: [1m[33m2.8484[0m
    || ITR: 32000 | Loss: [1m[33m2.8282[0m
    || ITR: 33000 | Loss: [1m[33m2.8762[0m
    || ITR: 34000 | Loss: [1m[33m2.8475[0m
    updated.... average c-index = 0.8863
    || ITR: 35000 | Loss: [1m[33m2.7535[0m
    updated.... average c-index = 0.8863
    || ITR: 36000 | Loss: [1m[33m2.8090[0m
    updated.... average c-index = 0.8871
    || ITR: 37000 | Loss: [1m[33m2.8471[0m
    updated.... average c-index = 0.8877
    || ITR: 38000 | Loss: [1m[33m2.7553[0m
    || ITR: 39000 | Loss: [1m[33m2.8352[0m
    || ITR: 40000 | Loss: [1m[33m2.8102[0m
    updated.... average c-index = 0.8884
    || ITR: 41000 | Loss: [1m[33m2.8306[0m
    updated.... average c-index = 0.8890
    || ITR: 42000 | Loss: [1m[33m2.7914[0m
    updated.... average c-index = 0.8894
    || ITR: 43000 | Loss: [1m[33m2.8027[0m
    updated.... average c-index = 0.8901
    || ITR: 44000 | Loss: [1m[33m2.7643[0m
    updated.... average c-index = 0.8905
    || ITR: 45000 | Loss: [1m[33m2.7775[0m
    || ITR: 46000 | Loss: [1m[33m2.8143[0m
    updated.... average c-index = 0.8916
    || ITR: 47000 | Loss: [1m[33m2.7644[0m
    updated.... average c-index = 0.8920
    || ITR: 48000 | Loss: [1m[33m2.7600[0m
    || ITR: 49000 | Loss: [1m[33m2.7252[0m
    updated.... average c-index = 0.8921
    || ITR: 50000 | Loss: [1m[33m2.7906[0m
    updated.... average c-index = 0.8928
    


```python
_, num_Event, num_Category  = np.shape(mask1)  # dim of mask1: [subj, Num_Event, Num_Category]

if num_Event > 1:
    in_path = data_mode + '/results_cr/'
else:
    in_path = data_mode + '/results/'

if not os.path.exists(in_path):
    os.makedirs(in_path)


FINAL1 = np.zeros([num_Event, len(eval_times), OUT_ITERATION])
FINAL2 = np.zeros([num_Event, len(eval_times), OUT_ITERATION])


for out_itr in range(OUT_ITERATION):
    in_hypfile = in_path + '/itr_' + str(out_itr) + '/hyperparameters_log.txt'
    in_parser = load_logging(in_hypfile)


    ##### HYPER-PARAMETERS
    mb_size                     = in_parser['mb_size']

    iteration                   = in_parser['iteration']

    keep_prob                   = in_parser['keep_prob']
    lr_train                    = in_parser['lr_train']

    h_dim_shared                = in_parser['h_dim_shared']
    h_dim_CS                    = in_parser['h_dim_CS']
    num_layers_shared           = in_parser['num_layers_shared']
    num_layers_CS               = in_parser['num_layers_CS']

    if in_parser['active_fn'] == 'relu':
        active_fn                = tf.nn.relu
    elif in_parser['active_fn'] == 'elu':
        active_fn                = tf.nn.elu
    elif in_parser['active_fn'] == 'tanh':
        active_fn                = tf.nn.tanh
    else:
        print('Error!')


    initial_W                   = tf.contrib.layers.xavier_initializer()

    alpha                       = in_parser['alpha']  #for log-likelihood loss
    beta                        = in_parser['beta']  #for ranking loss
    gamma                       = in_parser['gamma']  #for RNN-prediction loss
    parameter_name              = 'a' + str('%02.0f' %(10*alpha)) + 'b' + str('%02.0f' %(10*beta)) + 'c' + str('%02.0f' %(10*gamma))


    ##### MAKE DICTIONARIES
    # INPUT DIMENSIONS
    input_dims                  = { 'x_dim'         : x_dim,
                                    'num_Event'     : num_Event,
                                    'num_Category'  : num_Category}

    # NETWORK HYPER-PARMETERS
    network_settings            = { 'h_dim_shared'         : h_dim_shared,
                                    'h_dim_CS'          : h_dim_CS,
                                    'num_layers_shared'    : num_layers_shared,
                                    'num_layers_CS'    : num_layers_CS,
                                    'active_fn'      : active_fn,
                                    'initial_W'         : initial_W }


    # for out_itr in range(OUT_ITERATION):
    print ('ITR: ' + str(out_itr+1) + ' DATA MODE: ' + data_mode + ' (a:' + str(alpha) + ' b:' + str(beta) + ' c:' + str(gamma) + ')' )
    ##### CREATE DEEPFHT NETWORK
    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = Model_DeepHit(sess, "DeepHit", input_dims, network_settings)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    ### TRAINING-TESTING SPLIT
    (tr_data,te_data, tr_time,te_time, tr_label,te_label, 
     tr_mask1,te_mask1, tr_mask2,te_mask2)  = train_test_split(dhdata, time, label, mask1, mask2, test_size=0.20, random_state=seed) 

    (tr_data,va_data, tr_time,va_time, tr_label,va_label, 
     tr_mask1,va_mask1, tr_mask2,va_mask2)  = train_test_split(tr_data, tr_time, tr_label, tr_mask1, tr_mask2, test_size=0.20, random_state=seed) 
    
    ##### PREDICTION & EVALUATION
    saver.restore(sess, in_path + '/itr_' + str(out_itr) + '/models/model_itr_' + str(out_itr))

    ### PREDICTION
    pred = model.predict(te_data)
    
    ### EVALUATION
    result1, result2 = np.zeros([num_Event, len(eval_times)]), np.zeros([num_Event, len(eval_times)])

    for t, t_time in enumerate(eval_times):
        eval_horizon = int(t_time)

        if eval_horizon >= num_Category:
            print( 'ERROR: evaluation horizon is out of range')
            result1[:, t] = result2[:, t] = -1
        else:
            # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
            risk = np.sum(pred[:,:,:(eval_horizon+1)], axis=2) #risk score until EVAL_TIMES
            for k in range(num_Event):
                # result1[k, t] = c_index(risk[:,k], te_time, (te_label[:,0] == k+1).astype(float), eval_horizon) #-1 for no event (not comparable)
                # result2[k, t] = brier_score(risk[:,k], te_time, (te_label[:,0] == k+1).astype(float), eval_horizon) #-1 for no event (not comparable)
                result1[k, t] = weighted_c_index(tr_time, (tr_label[:,0] == k+1).astype(int), risk[:,k], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
                result2[k, t] = weighted_brier_score(tr_time, (tr_label[:,0] == k+1).astype(int), risk[:,k], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)

    FINAL1[:, :, out_itr] = result1
    FINAL2[:, :, out_itr] = result2

    ### SAVE RESULTS
    row_header = []
    for t in range(num_Event):
        row_header.append('Event_' + str(t+1))

    col_header1 = []
    col_header2 = []
    for t in eval_times:
        col_header1.append(str(t) + 'months c_index')
        col_header2.append(str(t) + 'months B_score')

    # c-index result
    df1 = pd.DataFrame(result1, index = row_header, columns=col_header1)
    df1.to_csv(in_path + '/result_CINDEX_itr' + str(out_itr) + '.csv')

    # brier-score result
    df2 = pd.DataFrame(result2, index = row_header, columns=col_header2)
    df2.to_csv(in_path + '/result_BRIER_itr' + str(out_itr) + '.csv')

    ### PRINT RESULTS
    print('========================================================')
    print('ITR: ' + str(out_itr+1) + ' DATA MODE: ' + data_mode + ' (a:' + str(alpha) + ' b:' + str(beta) + ' c:' + str(gamma) + ')' )
    print('SharedNet Parameters: ' + 'h_dim_shared = '+str(h_dim_shared) + ' num_layers_shared = '+str(num_layers_shared) + 'Non-Linearity: ' + str(active_fn))
    print('CSNet Parameters: ' + 'h_dim_CS = '+str(h_dim_CS) + ' num_layers_CS = '+str(num_layers_CS) + 'Non-Linearity: ' + str(active_fn)) 

    print('--------------------------------------------------------')
    print('- C-INDEX: ')
    print(df1)
    print('--------------------------------------------------------')
    print('- BRIER-SCORE: ')
    print(df2)
    print('========================================================')


    
### FINAL MEAN/STD
# c-index result
df1_mean = pd.DataFrame(np.mean(FINAL1, axis=2), index = row_header, columns=col_header1)
df1_std  = pd.DataFrame(np.std(FINAL1, axis=2), index = row_header, columns=col_header1)
df1_mean.to_csv(in_path + '/result_CINDEX_FINAL_MEAN.csv')
df1_std.to_csv(in_path + '/result_CINDEX_FINAL_STD.csv')

# brier-score result
df2_mean = pd.DataFrame(np.mean(FINAL2, axis=2), index = row_header, columns=col_header2)
df2_std  = pd.DataFrame(np.std(FINAL2, axis=2), index = row_header, columns=col_header2)
df2_mean.to_csv(in_path + '/result_BRIER_FINAL_MEAN.csv')
df2_std.to_csv(in_path + '/result_BRIER_FINAL_STD.csv')


### PRINT RESULTS
print('========================================================')
print('- FINAL C-INDEX: ')
print(df1_mean)
print('--------------------------------------------------------')
print('- FINAL BRIER-SCORE: ')
print(df2_mean)
print('========================================================')
```

    ITR: 1 DATA MODE: mortgage (a:1.0 b:3.0 c:0)
    INFO:tensorflow:Restoring parameters from mortgage/results_cr//itr_0/models/model_itr_0
    ========================================================
    ITR: 1 DATA MODE: mortgage (a:1.0 b:3.0 c:0)
    SharedNet Parameters: h_dim_shared = 300 num_layers_shared = 1Non-Linearity: <function relu at 0x1a39d3c8c8>
    CSNet Parameters: h_dim_CS = 300 num_layers_CS = 3Non-Linearity: <function relu at 0x1a39d3c8c8>
    --------------------------------------------------------
    - C-INDEX: 
             3months c_index  7months c_index  17months c_index
    Event_1         0.924171         0.905674          0.850083
    Event_2         0.924418         0.898984          0.846364
    --------------------------------------------------------
    - BRIER-SCORE: 
             3months B_score  7months B_score  17months B_score
    Event_1         0.259576         0.411762          0.462486
    Event_2         0.251459         0.370268          0.414376
    ========================================================
    ========================================================
    - FINAL C-INDEX: 
             3months c_index  7months c_index  17months c_index
    Event_1         0.924171         0.905674          0.850083
    Event_2         0.924418         0.898984          0.846364
    --------------------------------------------------------
    - FINAL BRIER-SCORE: 
             3months B_score  7months B_score  17months B_score
    Event_1         0.259576         0.411762          0.462486
    Event_2         0.251459         0.370268          0.414376
    ========================================================
    


```python
norm_mode='standard'
label              = np.asarray(data_sr[['event']])
time               = np.asarray(data_sr[['duration']])
dhdata             = np.asarray(data_sr.iloc[:,1:18])
dhdata             = f_get_Normalization(dhdata, norm_mode)

num_Category       = int(np.max(time) * 1.2)  #to have enough time-horizon
num_Event          = int(len(np.unique(label)) - 1) #only count the number of events (do not count censoring as an event)

x_dim              = np.shape(dhdata)[1]

mask1              = f_get_fc_mask2(time, label, num_Event, num_Category)
mask2              = f_get_fc_mask3(time, -1, num_Category)

DIM                = (x_dim)
DATA               = (dhdata, time, label)
MASK               = (mask1, mask2)

_EPSILON = 1e-08
seed = 1234
```


```python
x = tf.placeholder(tf.float32, [None, x_dim])
```


```python
w_init = tf.contrib.layers.xavier_initializer()
w_reg  = tf.contrib.layers.l2_regularizer(scale=1.0)
shared_out=FC_Net(x, 300, activation_fn=tf.nn.relu)
last_x = x  #for residual connection

h = tf.concat([last_x, shared_out], axis=1)
```


```python
h
```




    <tf.Tensor 'concat_1:0' shape=(?, 317) dtype=float32>




```python
sess = tf.Session()
```


```python
tf.print(h)
```




    <tf.Operation 'PrintV2' type=PrintV2>




```python

```


```python

```

---

# 6. Evaluation<a class="anchor" id="six"></a>


```python

```


```python

```


```python

```

---

# 7. Conclusion<a class="anchor" id="seven"></a>


```python

```


```python

```


```python

```

# 8. References<a class="anchor" id="eight"></a>
