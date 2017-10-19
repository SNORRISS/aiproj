# -*- coding: utf-8 -*-

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


from pylab import *
import pandas as pd
import numpy as np
import os
import matplotlib
import gc


from scipy import stats, integrate

import matplotlib.pyplot as plt




import seaborn as sns
sns.set(color_codes=True)
#sns.set_context('poster')
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

import sklearn.cluster as cluster
import time




lapd_data = pd.read_csv('/home/sam/Downloads/Arrest_Data_from_2010_to_Present.csv')
nsduh_data = pd.read_csv('/home/sam/Downloads/35509-0001-Data.tsv', sep = '\t')
nsduh_addy = nsduh_data[['NEWRACE2','ADDERALL']]
lapd_charge = lapd_data[['Charge','Descent Code']]

#lapd_charge = lapd_data[['Report ID','Charge']]
#nsduh_addy = nsduh_data[['CASEID','ADDERALL']]





