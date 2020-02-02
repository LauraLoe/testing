+++ 
title = "Deep Learning for Survival analysis" 
date = '2020-02-01' 
tags = [ "Deep Learning", "Neural Networks",]
categories = ["course projects"] 
author = "Seminar Information Systems WS19/20 - Laura Löschmann, Daria Smorodina" 
banner = "img/seminar/sample/hu-logo.jpg"
disqusShortname = "https-humbodt-wi-github-io-blog" 
description = "Introduction to Survival Analysis and following research with Neural Networks"
+++


# Deep Learning for Survival Analysis
---
#### Authors: Laura Löschmann, Daria Smorodina

## Table of content
1. [Motivation](#motivation) <br/>
2. [Basics of Survival Analysis](#introduction_sa)<br/>
* 2.1 [Common terms](#terms) <br />
* 2.2 [Survival function](#survival_function) <br />
* 2.3 [Hazard function](#hazard_function) <br />
3. [Dataset](#dataset) <br />
4. [Standard Methods in Survival Analysis](#standard_methods) <br />
* 4.1 [Kaplan-Meier estimate](#kmf) <br />
* 4.2 [Cox Proportional Hazard Model](#coxph) <br />
* 4.3 [Time-Varying Cox Regression Model](#time_cox) <br />
* 4.4 [Random Survival Forests](#rsf) <br />
5. [Deep Learning in Survival Analysis](#deeplearning_sa) <br />
* 5.1 [DeepSurv](#deepsurv) <br />
* 5.2 [Deep Hit](#deephit) <br />
6. [Evaluation](#evaluation)  <br />
7. [Conclusion](#conclusion) <br />
8. [References](#references) <br />

---

# 1. Motivation - Business case <a class="anchor" id="motivation"></a>
With the financial crisis hitting the United States and Europe in 2008, the International Accounting Standards Board (IASB) decided to revise their accounting standards for financial instruments, e.g. loans or mortgages to address perceived deficiencies which were believed to have contributed to the magnitude of the crisis.The result was the International Financial Reporting Standard (IFRS) 9 that became effective for all financial years beginning on or after 1 January 2018. [1]

Previously impairment losses on financial assets were only recognised to the extent that there was an objective evidence of impairment, meaning a loss event needed to occur before an impairment loss could be booked. [2] The new accounting rules for financial instruments require banks to build provisions for expected losses in their loan portfolio. The loss allowance has to be recognised before the actual credit loss is incurred. It is a more forward-looking approach than its predecessor with the aim to result in a more timely recognition of credit losses. [3]

To implement the new accounting rules banks need to build models that can evaluate a borrower's risk as accurately as possible.  A key credit risk parameter is the probability of default. Classification techniques such as logistic regression and decision trees can be used in order to classify the risky from the non-risky loans. These classification techniques however do not take the timing of default into account. With the use of survival analysis more accurate credit risks calculations are enabled since these analysis refers to a set of statistical techniques that is able to estimate the time it takes for a customer to default.

---

# 2. Introduction to Survival Analysis <a class="anchor" id="introduction_sa"></a>

Survival analysis also called time-to event analysis refers to the set of statistical analyses that takes a series of observations and attempts to estimate the time it takes for an event of interest to occur. 

The development of survival analysis dates back to the 17th century with the first life table ever produced by English statistician John Graunt in 1662. The name ‚Survival Analysis‘ comes from the longstanding application of these methods since throughout centuries they were solely linked to investigating mortality rates. However, during the last decades the applications of the statistical methods of survival analysis have been extended beyond medical research to other fields. [4]

Survival Analysis can be used in the field of health insurance to evaluate insurance premiums. It can be a useful tool in customer retention e.g. in order to estimate the time a customer probably will discontinue its subscription. With this information the company can intervene with some incentives early enough to retain its customer. The accurate prediction of upcoming churners results in highly-targeted campaigns, limiting the resources spent on customers who likely would have stayed anyway.
The methods of survival analysis can also be applied in the field of engineering, e.g. to estimate the remaining useful life of machines.

---

## 2.1 Common terms <a class="anchor" id="terms"></a>
Survival analysis is a collection of data analysis methods with the outcome variable of interest ‚time to event‘. In general ‚event‘ describes the event of interest, also called **death event**, ‚time‘ refers to the point of time of first observation, also called **birth event**, and ‚time to event‘ is the **duration** between the first observation and the time the event occurs. [5]
The subjects whose data were collected for survival analysis usually do not have the same time of first observation. A subject can enter the study at any time. Using durations ensure a necessary relativeness. [6] Referring to the business case the birth event is the initial recognition of a loan, the death event, consequently the event of interest, describes the time a customer defaulted and the duration is the time between the initial recognition and the event of default.

During the observation time not every subject will experience the event of interest. Consequently it is unknown if the subjects will experience the event of interest in the future. The computation of the duration, the time from the first observation to the event of interest, is impossible. This special type kind of missing data can emerge due to two reasons:

1. The subject is still part of the study but has not experienced the event of interest yet.
2. The subject experienced a different event which also led to the end of study for this subject.

In survival analysis this missing data is called **censorship** which refers to the inability to observe the variable of interest for the entire population. However, the censoring of data must be taken into account, dropping unobserved data would underestimate customer lifetimes and bias the results. Hence the particular subjects are labelled ‚censored‘.
Since for the censored subjects the death event could not be observed, the type of censorship is called right censoring which is the most common one in survival analysis. As opposed to this there is left censoring in case the birth event could not be observed. 

The first reason for censored cases regarding the use case are loans that have not matured yet and did not experience default by this time at the the moment of data gathering.

The second reason for censorship refers to loans that did not experience the event of default but the event of early repayment. With this the loan is paid off which results in the end of observation for this loan. This kind of censoring is used in models with one event of interest. [7]

In terms of different application fields an exact determination of the birth and death event is vital.
Following there are a few examples of birth and death events as well as possible censoring cases, besides the general censoring case that the event of interest has not happened yet, for various use cases in the industry:

Application field | Birth event | Death event | Censoring example
------------------|-------------|-------------|------------------
Predictive maintenance in mechanical operations|Time the machine was started for a continuous use|Time when the machine breaks down|Machine breaks down due to a fire in the factory building
Customer analytics|Customer starts subscription|Time when customer unsubscribe|Customer dies during the observation time
Medical research on breast cancer patients|Time the subject was first diagnosed|Time the patient died due to breast cancer|Patient died due to a cardiovasculae disease
Lifetimes of political leaders around the world|Start of the tenure|Retirement|Leader dies during the tenure

---

## 2.2 Survival Function<a class="anchor" id="survival_function"></a>
The set of statistic methods related to survival analysis has the goal to estimate the survival function from survival data. The survival function S(t) defines the probability that a subject of interest will survive beyond time t, or equivalently, the probability that the duration will be at least t. [8] The survival function of a population is defined as follows:

$$S(t) = Pr(T > t)$$

T is the random lifetime taken from the population under study and cannot be negative. With regard to the business case it is the amount of time a customer is able to pay his loan rates, he is not defaulting. The survival function S (t) outputs values between 0 and 1 and is a non-increasing function of t.
At the start of the study (t=0), no subject has experienced the event yet. Therefore the probability S(0) of surviving beyond time 0 is one. S(∞) =0 since if the study period were limitless, presumably everyone eventually would experience the event of interest and the probability of surviving would ultimately fall to zero. In theory the survival function is smooth, in practice the events are observed on a concrete time scale, e.g. days, weeks, months, etc., such that the graph of the survival function is like a step function. [9]

![survival_function](/blog/img/seminar/group2_SurvivalAnalysis/survival_function.png)

---

## 2.3 Hazard Function<a class="anchor" id="hazard_function"></a>
Derived from the survival function the hazard function h(t) gives the probability of the death event occurring at time t, given that the subject did not experience the death event until time t. It describes the instantaneous potential per unit time for the event to occur. [10]

$$h(t) = \lim_{\delta t\to 0}\frac{Pr(t≤T≤t+\delta t | T>t)}{\delta t}$$

Therefore the hazard function models which periods have the highest or lowest chances of an event. In contrast to the survival function, the hazard function does not have to start at 1 and go down to 0. The hazard rate usually changes over time. It can start anywhere and go up and down over time. For instance the probability of defaulting on a mortgage may be low in the beginning but can increase over the time of the mortgage.

![hazard_function](/blog/img/seminar/group2_SurvivalAnalysis/hazard_function.png)

The above shown graph is a theoretical example for a hazard function. [11] This specific hazard function is also called bathtub curve due to its form. This graph shows the probability of an event of interest to occur over time. 

It could describe the probability of a customer unsubscribing from a magazine over time. Within the first 30 days the risk to unsubscribe is high, since the customer is testing the product. But if the customer likes the content, meaning he "survives" the first 30 days, the risk of unsubscribing decreased and stagnates at lower level. After a while the risk is increasing again since the customer maybe needs different input or got bored over time. Hence the graph gives the important information when to initiate incentives for those customers whose risk to unsubsribe is about to increase in order to retain them.

The main goal of survival analysis is to estimate and interpret survival and/or hazard functions from survival data. 
 
---

# 3. Dataset <a class="anchor" id="dataset"></a>

We used the real-world dataset of 50.000 US mortgage borrowers which was provided by International Financial Research (www.internationalfinancialresearch.org). 
The data is given in a "snapshot" panel format and represents a collection of US residential mortgage portfolios over 60 periods. Loan can originate before the initial start of our study and paid after it will be finished as well.

When a person applies for mortgage lenders (banks) want to know value of risk they would take by loaning money. 
In the given dataset we are able to inspect this process using the key information from following features:
- various timestamps for loan origination, future maturity and first appearance in the survival study
- outside factors like gross domestic product (GDP) or unemployment rates at observation time
- average price index at observation moment
- FICO score for each individual: the higher the score, the lower the risk (a "good" credit score is considered to be in the 670-739 score range)
- interest rates for every issued loan
- since our object of analysis is a mortgage data we have some insights for inquired real estate types (home for a single family or not, is this property in area with urban development etc.) which also are playing an important role for loan amount.

While interpreting our data as for survival analysis the *birth event* is the time when the subject was first observed for the study and the *death event* is the default of the subject. The duration is the time between the birth and death event. The dataset does not contain any lost or withdrawn subjects but there exist subjects who have not defaulted yet. These subjects will be labelled *censored* in further analysis.

The graph below shows an example for censorship concept exactly for the given mortgage dataset at specific timepoint (in this case 13 months).  

![censorship_plot](/blog/img/seminar/group2_SurvivalAnalysis/censorship.png)

Some individuals defaulted before this time and the rest either continue their lifetime and experience the event later by close of study or different event occurs (right-censoring).
This leads to one of the challenges in survival analysis: how to handle properly this information. In general distribution of event of interest (in graph below) more than 2/3 of inspected individuals labeled as "censored" and dropping out these observations will lead to significant information loss and biased outcome. Since the survival analysis was developed to solve this problem, all given values would be taken for further research.

![event_distrib_plot](/blog/img/seminar/group2_SurvivalAnalysis/event_distrib.png)

Further computation for survival analysis requires a specific dataset format: *total_obs_time* column represents calculated lifetime duration for each borrower (for censored objects it would be a study time, for defaulted - time taken before event happening), *default_time* corresponds to event indicator (1 for experienced and 0 in case of censoring) and *X* - p-dimensional feature vector.

![data_snapshot](/blog/img/seminar/group2_SurvivalAnalysis/subset_data.png)

---

# 4. Standard Methods in Survival Analysis<a class="anchor" id="standard_methods_sa"></a>

The standard ways for estimation can be classified into three main groups: **non-parametric**, **semi-parametric**, and **parametric** approaches. The choice which method to use should be guided by dataset design and research question of interest. It is feasible to use sometimes more than one approach.

- **Parametric** methods rely on assumptions that distribution of the survivaltimes corresponds to specific probability distributions. This group consists of methods such as exponential, Weibull and lognormal distributions. Parameters inside these models are usually estimated using certain maximum likelihood estimations.
- In the **non-parametric** methods there are no dependencies on the form of parameters in underlying distributions. Mostly, the non-parametric approach is used to describe survival probabilities as function of time and to give an average view of individual's population. The most popular univariate method is **Kaplan-Meier estimate** and used as first step in survival descriptive analysis (section 4.1).
- To the **semi-parametric** methods correspond the **Cox regression model** which is based both on parametric and non-parametric components (section 4.2).

The easiest way to estimate the survival function using aforementioned approaches is a Python package *lifelines* (available for all operating systems at https://lifelines.readthedocs.io/en/latest/).

## 4.1 Kaplan Meier Estimate<a class="anchor" id="kmf"></a>

The key idea of Kaplan-Meier estimator is to break the estimation of survival function $S(t)$ into a smaller steps depending on observed event times. For each interval the probability of surviving until the end of this interval is calculated, given the following formula:

$$ \hat{S(t)} = \prod_{i: t_i <= t}{\frac{n_i - d_i}{n_i}} ,$$
where $n_i$ is a number of individuals who are at risk at time point $t_i$ and $d_i$ is a number of subjects with experienced event at time $t_i$.

When using Kaplan-Meier Estimate, some assumptions must be taken into account:
- All observations - both censored and with event of interest - are used in estimation
- There is no cohort effect on survival, so subjects have the same survival probability regardless of their nature and time of appearance in study
- Individuals who are censored have the same survival probabilities as those who continue to be examined
- Survival probability is equal for all subjects.

The main disadvantage of this method is that it cannot estimate survival probability considering all covariates in the data (it is an *univariate* approach). That's why semi- and parametric models allow to analyse survival data and estimate $S(t)$ with respect to them.

```python
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()

T = data["total_obs_time"]
E = data["default_time"]

kmf.fit(T, event_observed = E)
```
The estimated $S(t)$ can be plotted as a stepwise function of all population-individuals and gives a nice way to make a visualization of survival experience.

![kmf_plot](/blog/img/seminar/group2_SurvivalAnalysis/kmf.png)

---

## 4.2 Cox Proportional Hazard Model<a class="anchor" id="coxph"></a>


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




![png](output_97_1.png)


### Plotting the effect of varying a covariate

After fitting, we can plot what the survival curves look like as we vary a single covariate while holding everything else equal. This is useful to understand the impact of a covariate, given the model. To do this, we use the plot_covariate_groups() method and give it the covariate of interest, and the values to display.


```python
cph.plot_covariate_groups('real_estate_single_family_home', values=range(0, 15, 3), cmap='coolwarm',
                         plot_baseline=True)

plt.title("The CoxPH Model for Variable: Real Estate Single Family Home", fontsize = 15)
plt.ylabel("Probability a Borrower is not defaulted", fontsize = 15)

plt.show()
```


![png](output_100_0.png)



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




![png](output_102_1.png)



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

## 4.3 Time-Varying Cox Regression Model<a class="anchor" id="time_cox"></a>


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




![png](output_136_1.png)



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

## 4.4 Random Survival Forests<a class="anchor" id="rsf"></a>


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

# 5. Deep Learning in Survival Analysis<a class="anchor" id="deeplearning_sa"></a>

## 5.1 DeepSurv<a class="anchor" id="deepsurv"></a>


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


![png](output_182_0.png)



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


![png](output_188_0.png)



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


![png](output_194_0.png)



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


![png](output_199_0.png)



```python
ev.integrated_brier_score(time_grid)
```




    0.11487176920079373




```python
ev.integrated_nbll(time_grid)
```




    1.6275194724001085



---

## 5.2 DeepHit<a class="anchor" id="deephit"></a> 

The model called „DeepHit“ was introduced in a paper by Changhee Lee, William R. Zame, Jinsung Yoon, Mihaela van der Schaar in April 2018. It describes a deep learning approach to survival analysis implemented in a tensor flow environment.

DeepHit is a deep neural network that learns the distribution of survival times directly. This means that this model does not do any assumptions about an underlying stochastic process, so both the parameters of the model as well as the form of the stochastic process depends on the covariates of the specific dataset used for survival analysis. [x]

The model basically contains two parts, a shared sub-network and a family of cause-specific sub-networks. Due to this architecture a great advantage of DeepHit is that it easily can be used for survival datasets with one single risk but also with multiple competing risks.
The dataset used so far describes one single risk, the risk of default. Customers that did not experience the event of interest are censored. The reasons for censorship can either be that the event of interest was not experienced or another event happened that also led to the end of observation, but is not the event of interest for survival analysis. 
The original dataset has information about a second risk, the early repayment, also called payoff. For prior use the dataset was preprocessed in a way that customers with an early repayment were also labelled „censored“, because the only event of interest was the event of default. If the second risk also becomes the focus of attention in terms of survival analysis a second label for payoff (payoff = 2) can be introduced in the event column of the dataset. Therefore a competing risk is an event whose occurrence precludes the occurrence of the primary event of interest. [b]


```python
data_cr = df.copy()
```


```python
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



To also handle competing risks DeepHit provides a flexible multi-task learning architecture.
Multi-task learning was originally inspired by human learning activities. People often apply the  knowledge learned from previous tasks to help learn a new task. For example, for a person who learns to ride the bicycle and unicycle together, the experience in learning to ride a bicycle can be utilized in riding a unicycle and vice versa. Similar to human learning, it is useful for multiple learning tasks to be learned jointly since the knowledge contained in a task can be leveraged by other tasks. 
In the context of deep learning models, multiple models could be trained, each model only learning one tasks (a). If this multiple tasks are related to each other, a multi-task learning model can be used with the aim to improve the learning of a model by using the knowledge achieved throughout the learning of related tasks in parallel (b). [y] 

![mult_task1](/blog/img/seminar/group2_SurvivalAnalysis/multitasking_1.png)

Multi-task learning is similar to transfer learning but has some significant differences. Transfer learning models use several source tasks in order to improve the performance on the target task. Multi-task learning models treat all tasks equally, there is no task importance hierarchy. There is no attention focus on one specific task. The goal of multi-task learning models is to improve the performance of all tasks.

The most commonly used approach to multi-task learning in neural networks is called hard parameter sharing. The general architecture of such a multi-task learning model describes two main parts. The first part is a shared sub-network, where the model learns the common representation of the related tasks. The model then splits into task-specific sub-networks in order to learn the non-common parts of the representation. The number of task-specific sub-networks is equal to the number of related tasks the model is trained on.
For the sake of completeness another approach to multi-task learning is soft parameter sharing that describes an architecture where each task has its own model with its own parameters. To encourage the parameters to become similar regularisation techniques are applied between the parameters of the task-specific models. Since DeepHit provides an architecture of hard parameter sharing, the approach of soft parameter sharing will be neglected in further explanations.

![mult_task2](/blog/img/seminar/group2_SurvivalAnalysis/multitasking_2.png)

To train a multi-task learning model just as many loss functions as tasks are required. The model is then trained by backpropagation. The fact that the task-specific sub-networks share common hidden layers, allows comprehensive learning. Through the shared hidden layers features that are developed in the hidden layers of one task can also be used by other tasks. Multi-task learning enables features to be developed to support several tasks which would not be possible if multiple singe-task learning models would be trained on the related tasks in isolation. Also some hidden units can specialise on one task, providing information that are not important for the other tasks. By keeping the weights to these hidden units small gives these tasks the opportunity to ignore these hidden units. [z] 

With multi-task learning a model can increase its performance due to several reasons. By using the data of multiple related tasks multi-task learning increases the sample size that is used to train the model which is a kind of implicit data augmentation. The network sees more labels, even though these labels are not the labels from the same task but highly related tasks. A model that learns different similar tasks simultaneously is able to learn a more general representation that captures all of the tasks.
Moreover by learning multiple tasks together the network has to focus on important information rather than task-specific noise. The other tasks provide additional evidence for the relevance or irrelevance of the features and help to attract the network´s attention to focus on the important features.
Some tasks are harder to learn even by themselves. A model can benefit from learning the hard task combined with an easier related task. Multi-task learning allows the model to eavesdrop, learn the hard task through the simple related task, and therefore learn the hard task easier and faster than learning the hard task in isolation. 
In addition different related tasks can treat each other as a form of regularisation term since the model has to learn a general representation of all tasks. Learning the tasks in a single-task learning approach would bear the risk of overfitting on one task. [a] 

![deephit](/blog/img/seminar/group2_SurvivalAnalysis/deephit.png)

The architecture of the DeepHit model is similar to the conventional multi-task learning architecture of hard parameter sharing, but has two main differences. DeepHit provides a residual connection between the original covariates and the input of the cause-specific sub-networks. This means that the input of the cause-specific sub-networks is not only the output of the preceded shared sub-network but also the original covariates. These additional input allows the cause-specific sub-network to better learn the non-common representation of the multiple causes.
The other difference refers to the final output of the model. DeepHit uses one single softmax output layer so that the model can learn the joint distribution of the competing events instead of their marginal distribution. Thus the output of the DeepHit model is a vector for every subject in the dataset giving the probabilities that the subject with covariates x will experience the event k for every timestamp t within the observation time. The probabilities of one subject sum up to 1.

$$y = [y_{1,1},...,y_{1,Tmax},...,y_{K,1},...,y_{K,Tmax}]$$


The visualisation of the DeepHit model shows the architecture for a survival dataset of two competing risks. This architecture can easily be adjusted to more or less competing risks by adding   or removing cause-specific sub-networks. The architecture of the DeepHit model depends on the number of risks.

To implement the model the [DeepHit repository](https://github.com/chl8856/DeepHit) has to be cloned to create a local copy on the computer. The following packages need to be imported:


```python
from class_DeepHit import Model_DeepHit, log, div
import import_data as impt
from import_data import f_get_Normalization, f_get_fc_mask2, f_get_fc_mask3
import get_main
from get_main import f_get_minibatch 
from main_RandomSearch import save_logging, load_logging , get_random_hyperparameters
from utils_eval import c_index, brier_score, weighted_c_index, weighted_brier_score
from summarize_results import load_logging
import utils_network as utils

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected as FC_Net
from sklearn.model_selection import train_test_split
import random
import time, datetime, os
from termcolor import colored
_EPSILON = 1e-08
seed = 1234
```

DeepHit also needs the characteristic survival analysis input setting containing the event labels, the durations as well as the covariates. A function is provided that either applies standardisation or normalization of the data. For this analysis standardisation was applied on the data. 

The variable num_Category describes the dimension of the time horizon of interest and is needed in order to calculate the output dimension of the output layer of the model.
num_Event gives the number of events excluding the case of censoring, since censoring is not an event of interest. This number defines the architecture of the model, it specifies the number of cause-specific sub-networks and is also needed to calculate the dimension of the output layer, which is the multiplication of num_Category and num_Event.
The input dimension is defined by the number of covariates used to feed the network. 


```python
# DeepHit input settings

# Characteristic data format E, T, X
event              = np.asarray(data[['event']])
time               = np.asarray(data[['duration']])
dhdata             = np.asarray(data.iloc[:,1:18])

# Standardisation of the data
dhdata             = f_get_Normalization(dhdata, 'standard')

# Dimension of time horizon of interest (equivalent to the output dimension per risk)
num_Category       = int(np.max(time) * 1.2) 

# Number of events (censoring is not included)
num_Event          = int(len(np.unique(event)) - 1) 

# Input dimension
x_dim              = np.shape(dhdata)[1]

# Based on the data, mask1 and mask2 needed to calculate the loss functions
# To calculate loss 1 - log-likelihood loss
mask1              = f_get_fc_mask2(time, event, num_Event, num_Category)
# To calculate loss 2 - cause-specific ranking loss
mask2              = f_get_fc_mask3(time, -1, num_Category)

DIM                = (x_dim)
DATA               = (dhdata, time, event)
MASK               = (mask1, mask2)

```

The hyperparameters of DeepHit can be tuned by running random search using cross-validation. The function get_random_hyperparameters randomly takes values for parameters out of a predefined range for those parameters. 
Possible candidates for parameter tuning can be:

* Batch size
* Number of layers for the shared sub-network
* Number of layers for the cause-specific sub-network
* Number of nodes for the shared sub-network
* Number of nodes for the cause-specific sub-network
* Learning rate
* Dropout
* Activation function

The chosen parameters are forwarded to the function get_valid_performance along with the event labels, durations and covariates (summarized in DATA) as well as the masks for the loss calculations (summarized in MASK). This function takes the forwarded parameters to build a DeepHit model corresponding to the number of events of interest as well as the number of layers and nodes for the sub-networks. The dataset is then spilt into training, validation and test sets in order to start training the model on the training set using the chosen parameters. The training is done with mini batches of the training set over 50.000 iterations. Every 1000 iteration a prediction is done on the validation set and the best model is saved to the specified file path. The best result is returned if there is no improvement for the next 6000 iterations (early stopping).


```python
# Hyperparameter tuning

# Number of training/validation/test splits during tuning
OUT_ITERATION               = 1

# Number of random search iterations
RS_ITERATION                = 20

# For saving purposes of the best parameters
data_mode = 'mortgage'
out_path  = data_mode + '/results'

# Times when the validation is performed
eval_times = [4,8,18]

for itr in range(OUT_ITERATION):
    
    if not os.path.exists(out_path + '/itr_' + str(itr) + '/'):
        os.makedirs(out_path + '/itr_' + str(itr) + '/')

    max_valid = 0
    log_name = out_path + '/itr_' + str(itr) + '/hyperparameters_log.txt'

    for r_itr in range(RS_ITERATION):
        print('OUTER_ITERATION: ' + str(itr))
        print('Random search... itr: ' + str(r_itr))
        new_parser = get_random_hyperparameters(out_path)
        print(new_parser)

        # get validation performance given the hyperparameters
        tmp_max = get_valid_performance(DATA, MASK, new_parser, itr, eval_times, MAX_VALUE=max_valid)

        if tmp_max > max_valid:
            max_valid = tmp_max
            max_parser = new_parser
            save_logging(max_parser, log_name)  # save the hyperparameters if they provide the maximum validation performance

        print('Current best: ' + str(max_valid))
```

DeepHit is build with Xavier initialisation and dropout for all the layers and is trained by back propagation via the Adam optimizer. To train a survival analysis model like DeepHit a loss function has to be minimised that is especially designed to handle censored data.
The loss function of the DeepHit model is the sum of two terms. 

$$ L_{Total} = L_{1} + L_{2}$$

$L_{1}$ is the log-likelihood of the joint distribution of the first hitting time and event. This function is modified in a way that it captures censored data and considers competing risks if necessary. 
The log-likelihood function also consists out of two terms. The first term captures the event and the time, the event occurred, for the uncensored customers. The second term captures the time of censoring for the censored customers giving the information that the customer did not default up to that time.

$L_{2}$ is a combination of cause-specific ranking loss functions since DeepHit is a multi-task learning model and therefore needs cause-specific loss functions for training. The ranking loss function incorporates the estimated cumulative incidence function calculated at the time the specific event occurred. The formula of the cumulative incidence function (CIF) is as follows:

$$F_{k^{*}}(t^{*}|x^{*}) = \sum_{s^{*}=0}^{t^{*}}P(s=s^{*},k=k^{*}|x=x^{*})$$

This function expresses the probability that a particular event k occurs on or before time t conditional on covariates x. To get the estimated CIF, the sum of the probabilities from the first observation time to the time, the event k occurred, is computed.

$$ \hat{F}_{k^{*}}(s^{*}|x^{*}) = \sum_{m=0}^{s^{*}}y^{*}_{k,m}$$

The cause-specific ranking loss function adapts the idea of concordance. A customer that experienced the event k on a specific time t should have a higher probability than a customer that will experience the event sometime after this specific time t. The ranking loss function therefore compares pairs of customers that experienced the same event of interest and penalizes an incorrect ordering of pairs.

After the training process the saved optimised hyper-parameters as well as the corresponding trained model can be used for the final prediction on the test dataset.

```python
# Load the saved optimised hyperparameters
in_hypfile = in_path + '/itr_' + str(out_itr) + '/hyperparameters_log.txt'
in_parser = load_logging(in_hypfile)


# Forward the hyperparameters
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

# Create the dictionaries 
# For the input settings
input_dims                  = { 'x_dim'                : x_dim,
                                'num_Event'            : num_Event,
                                'num_Category'         : num_Category}

# For the hyperparameters
network_settings            = { 'h_dim_shared'         : h_dim_shared,
                                'h_dim_CS'             : h_dim_CS,
                                'num_layers_shared'    : num_layers_shared,
                                'num_layers_CS'        : num_layers_CS,
                                'active_fn'            : active_fn,
                                'initial_W'            : initial_W }

 
# Create the DeepHit network architecture
tf.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

model = Model_DeepHit(sess, "DeepHit", input_dims, network_settings)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

# Training, test sets split
(tr_data,te_data, tr_time,te_time, tr_label,te_label, 
 tr_mask1,te_mask1, tr_mask2,te_mask2)  = train_test_split(dhdata, time, event, mask1, mask2, test_size=0.20, random_state=seed) 
    
# Restoring the trained model
saver.restore(sess, in_path + '/itr_' + str(out_itr) + '/models/model_itr_' + str(out_itr))

## Final prediction on the test set covariates
pred = model.predict(te_data)
    


    ### EVALUATION
    result1, result2 = np.zeros([num_Event, len(eval_times)]), np.zeros([num_Event, len(eval_times)])

    for t, t_time in enumerate(eval_times):
        eval_horizon = int(t_time)

        if eval_horizon >= num_Category:
            print( 'ERROR: evaluation horizon is out of range')
            result1[:, t] = result2[:, t] = -1
        else:
            risk = np.sum(pred[:,:,:(eval_horizon+1)], axis=2) #risk score until EVAL_TIMES
            for k in range(num_Event):
                result1[k, t] = weighted_c_index(tr_time, (tr_label[:,0] == k+1).astype(int), risk[:,k], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) 
                result2[k, t] = weighted_brier_score(tr_time, (tr_label[:,0] == k+1).astype(int), risk[:,k], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) 

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

---

# 6. Evaluation<a class="anchor" id="evaluation"></a>

For the evaluation of survival analysis models the performance measures need to take censored data into account. The most common evaluation metric in survival analysis is the concordance index. It shows the model`s ability to correctly provide a reliable ranking of the survival times based on the individual risk scores. The idea behind concordance is that a subject that dies at time s should have a higher risk at time s than a subject who survives beyond time s. The concordance index expresses the proportion of concordant pairs in a dataset, thus estimates the probability that, for a random pair of individuals, the predicted survival times of the two individuals have the same ordering as their true survival times. A concordance index of 1 represents a model with perfect prediction, an index of 0.5 is equal to random prediction.
For a better understanding of this definition the concordance index is calculated on some simple example predictions. The following table shows the true default times of four theoretical customers along with default time predictions of three different models.


Ex. 1 | True default time | Model 1 | Model 2 |Model 3
:-----:|:------:|:-----:|:------:|:------:
Customer A|1|1|1|2
Customer B|2|2|2|3
Customer C|3|3|4|4
Customer D|4|4|3|5

To calculate the concordance index the number of concordant pairs has to be divided by the number of possible. By having four customers the following pairs are possible:
(A,B) , (A,C) , (A,D) , (B,C) , (B,D) , (C,D). The total number of possible pairs is 6. Model 1 predicts that A defaults before B, and the true default time confirms that A defaults before B. The pair (A,B) is a concordant pair. This comparison needs to be done for every possible pair. For the prediction of Model 1 all possible pairs are concordant, which results in an Concordance index of 1 - perfect prediction.
For the prediction of Model 2 there are five concordant pairs, but the for the pair (C,D) the model predicts that D defaults before C, whereas the true default times show that C defaults before D. With this the concordance index is 0.83 (5/6).
The concordance index of Model 3 is also equal to 1, since the model predicts the correct order of the possible pairs even though the actual default times are not right in isolation.

The next example shows the computation of the concordance index in case of right-censoring:

Ex. 2 | True default time|Model prediction|Censoring
:----:|:----:|:----:|:----:
Customer A|1|1|False
Customer B|2|2|True
Customer C|3|4|False
Customer D|4|3|False

The first step is to figure the number of possible pairs. The default times of customer A can be compared to the default times of the other customers. The customer B is censored, which means that the only information given is the fact that customer B did not default up to time 2, but there is no information if customer B will default and if so, when the customer will experience the event of default. Therefore a comparison between customer B and C as well as customer B and D is impossible because these customers defaulted after customer B was censored. The comparison between customers C and D is possible since both customers are not censored. In total there are four possible pairs: 
(A,B) , (A,C) , (A,D), (C,D)
The second step is to check if these possible pairs are concordant. The first three pairs are concordant, the pair (C,D) is discordant. The result is a concordance index of 0.75 (3/4).

The dataset used for the blogpost features the case of right-censoring but the reason for censoring is that these customers are still in the phase of repaying and their loans has not matured yet. Therefore the time of censoring is equal to the last observation time. Due to this the case that some customer default after a customer was censored is not possible. The example of the concordance index in case of right-censoring is shown for the sake of completeness since other survival datasets can have this case. A medical dataset for example can have data about patients with a heart disease. If a patient dies due to different reasons than a heart disease this patient would be censored. This can happen during the observation time and other patients can die due to a heart disease at a later time.


```python

```


```python

```


```python

```

---

# 7. Conclusion<a class="anchor" id="conclusion"></a>


```python

```


```python

```


```python

```

# 8. References<a class="anchor" id="references"></a>

[1] IFRS 9 Financial Instruments - https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/#about (accessed: 29.01.2020)

[2] Ernst & Young (December 2014): Impairment of financial instruments under IFRS 9 - https://www.ey.com/Publication/vwLUAssets/Applying_IFRS:_Impairment_of_financial_instruments_under_IFRS_9/$FILE/Apply-FI-Dec2014.pdf

[3] Bank for International Settlements (December 2017): IFRS 9 and expected loss provisioning - Executive Summary - https://www.bis.org/fsi/fsisummaries/ifrs9.pdf



[4] Liberato Camilleri (March 2019): History of survival snalysis - https://timesofmalta.com/articles/view/history-of-survival-analysis.705424



[5] Sucharith Thoutam (July 2016): A brief introduction to survival analysis

[6] Taimur Zahid (March 2019): Survival Analysis - Part A - https://towardsdatascience.com/survival-analysis-part-a-70213df21c2e 

[7] Lore Dirick, Gerda Claeskens, Bart Baesens (2016): Time to default in credit scoring using survival analysis: a benchmark study

[8] lifelines - Introduction to survival analysis - https://lifelines.readthedocs.io/en/latest/Survival%20Analysis%20intro.html

[9] Nidhi Dwivedi, Sandeep Sachdeva (2016): Survival analysis: A brief note - https://lifelines.readthedocs.io/en/latest/Survival%20Analysis%20intro.html

[10] Maria Stepanova, Lyn Thomas (2000): Survival analysis methods for personal loan data

[11] Hazard Function: Simple Definition - https://www.statisticshowto.datasciencecentral.com/hazard-function/ (accessed 29.01.2020)

[1]<a class="anchor" id="ref-one"></a> Jared L. Katzman, Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting Jiang,
and Yuval Kluger. DeepSurv: personalized treatment recommender system using a Cox
proportional hazards deep neural network. 2018.

[x] Changhee Lee, William R. Zame, Jinsung Yoon, Mihaela van der Schaar (April 2018): DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks

[y] Yu Zhang, Qiang Yang (2018): A survey on Multi-Task Learning

[z] Rich Caruana (1997): Multitask Learning

[a] Sebastian Rude (October 2017): An Overview of Multi-Task Learning in Deep Neural Networks

[[1]](#ref-one)    #### put in texts above as link to reference list
