#!/usr/bin/env python
# coding: utf-8

# Impact of Latino Migration on African American Labor and Wages in the US

# Varsha Vaidyanath, UC Berkeley

# Importing Data Science Libraries

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')

get_ipython().run_line_magic('matplotlib', 'inline')
from datascience import Table
from datascience.predicates import are
from datascience.util import *


# Importing Data as Table

general_tbl = Table.read_table("DemogFinalProject.csv")
general_tbl = general_tbl.drop("BPL", "BPLD", "YRIMMIG")
general_tbl


# Creating Tables Filtered by Year and Origin, Grouped By Metropolitan Area

grouped2000 = general_tbl.where("YEAR", are.equal_to(2000)).group("MET2013")
grouped2000 = grouped2000.where("MET2013", are.not_equal_to(0))
num_hispanic2000 = make_array()
af_wage2000 = make_array()
for i in np.arange(grouped2000.num_rows):
    tbl_specific = general_tbl.where("MET2013", are.equal_to(grouped2000.column("MET2013").item(i)))
    num_hispanic2000 = np.append(num_hispanic2000, (tbl_specific.where("HISPAN", are.not_equal_to(0)).num_rows)/tbl_specific.num_rows)
    af_tbl = tbl_specific.where("RACBLK", are.equal_to(2))
    af_wage2000 = np.append(af_wage2000, np.average(af_tbl.column("INCTOT")))

grouped2017 = general_tbl.where("YEAR", are.equal_to(2017)).group("MET2013")
grouped2017 = grouped2017.where("MET2013", are.not_equal_to(0))
num_hispanic2017 = make_array()
af_wage2017 = make_array()
for i in np.arange(grouped2017.num_rows):
    tbl_specific = general_tbl.where("MET2013", are.equal_to(grouped2017.column("MET2013").item(i)))
    num_hispanic2017 = np.append(num_hispanic2017, (tbl_specific.where("HISPAN", are.not_equal_to(0)).num_rows)/tbl_specific.num_rows)
    af_tbl = tbl_specific.where("RACBLK", are.equal_to(2))
    af_wage2017 = np.append(af_wage2017, np.average(af_tbl.column("INCTOT")))

tbl_2000 = Table().with_columns("Hispanic Percent 2000", num_hispanic2000, "African American Wages 2000",
                               af_wage2000)
tbl_2017 = Table().with_columns("Hispanic Percent 2017", num_hispanic2017, "African American Wages 2017",
                               af_wage2017)


# Equations for Regression

def std_u(arr):
    return (arr - np.mean(arr))/np.std(arr)

def find_r(tbl, col_x, col_y):
    return np.mean(std_u(tbl.column(col_x))*std_u(tbl.column(col_y)))

def slope(tbl, col_x, col_y):
    r = find_r(tbl, col_x, col_y)
    return r*np.std(tbl.column(col_y))/np.std(tbl.column(col_x))

def intercept(tbl, col_x, col_y):
    return np.mean(tbl.column(col_y)) - slope(tbl, col_x, col_y)*np.mean(tbl.column(col_x))


# Visualizing the Comparison of Hispanic Percentages to African American Wages in 2000 and 2017

#scatterplot2000 
line_2000 = (slope(tbl_2000, "Hispanic Percent 2000", "African American Wages 2000") * tbl_2000.column("Hispanic Percent 2000")) + intercept(tbl_2000, "Hispanic Percent 2000", "African American Wages 2000")
tbl_2000.scatter("Hispanic Percent 2000", "African American Wages 2000", fit_line = True)

#scatterplot2017
line_2017 = (slope(tbl_2017, 0, 1) * tbl_2017.column(0)) + intercept(tbl_2017, 0, 1)
tbl_2017.scatter("Hispanic Percent 2017", "African American Wages 2017", fit_line = True)


# Creating Bootstraps By Resampling

#bootstrap2000
bootstrap2000 = make_array()
for i in np.arange(1000):
    boot_samp00 = grouped2000.sample()
    resampled_slope = slope(boot_samp00, "MET2013", "count")
    bootstrap2000 = np.append(bootstrap2000, resampled_slope)

#bootstrap2017
bootstrap2017 = make_array()
for i in np.arange(1000):
    boot_samp17 = grouped2017.sample()
    resampled_slope = slope(boot_samp17, "MET2013", "count")
    bootstrap2017 = np.append(bootstrap2017, resampled_slope)
    

# Table of Bootstrap Differences in 2000 and 2017

bootstrap_tbl = Table().with_columns("Bootstrap 2000", bootstrap2000, "Bootstrap 2017", bootstrap2017,
                                    "Differences", bootstrap2017-bootstrap2000)
bootstrap_tbl


# Histogram of 95% Confidence Interval, Representing Bootstraps in 2000 and 2017

#bootstrap hist 2000
bootstrap_tbl.hist("Bootstrap 2000", unit="")

p05_2000=Table().with_column('dind',bootstrap_tbl.column("Bootstrap 2000")).percentile(2.5)['dind'][0]
p95_2000=Table().with_column('dind',bootstrap_tbl.column("Bootstrap 2000")).percentile(97.5)['dind'][0]
plt.axvline(x=p05_2000,color='green',linewidth=1)
plt.axvline(x=p95_2000,color='green',linewidth=1)

left_2000 = percentile(2.5, bootstrap_tbl.column("Bootstrap 2000"))
right_2000 = percentile(97.5, bootstrap_tbl.column("Bootstrap 2000"))
print("The 95% confidence interval for 2000 is:", 
     left_2000 , "and", right_2000)

#bootstrap hist 2017
bootstrap_tbl.hist("Bootstrap 2017", unit="")
p05_2017=Table().with_column('dind',bootstrap_tbl.column("Bootstrap 2017")).percentile(2.5)['dind'][0]
p95_2017=Table().with_column('dind',bootstrap_tbl.column("Bootstrap 2017")).percentile(97.5)['dind'][0]
plt.axvline(x=p05_2017,color='green',linewidth=1)
plt.axvline(x=p95_2017,color='green',linewidth=1)

left_2017 = percentile(2.5, bootstrap_tbl.column("Bootstrap 2017"))
right_2017 = percentile(97.5, bootstrap_tbl.column("Bootstrap 2017"))
print("The 95% confidence interval for 2017 is:", 
     left_2017 , "and", right_2017)


# Histogram of 95% Confidence Interval, Representing Difference Between 2000 and 2017 Bootstaps

#bootstrap hist differences
bootstrap_tbl.hist("Differences", unit="")
#confidence interval

p05_diff=Table().with_column('dind',bootstrap_tbl.column("Differences")).percentile(2.5)['dind'][0]
p95_diff=Table().with_column('dind',bootstrap_tbl.column("Differences")).percentile(97.5)['dind'][0]
plt.axvline(x=p05_diff,color='green',linewidth=1)
plt.axvline(x=p95_diff,color='green',linewidth=1)

left_diff = percentile(2.5, bootstrap_tbl.column("Differences"))
right_diff = percentile(97.5, bootstrap_tbl.column("Differences"))
print("The 95% confidence interval for the difference between the two data sets is:", 
     left_diff , "and", right_diff)
