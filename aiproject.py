# -*- coding: utf-8 -*-

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


from pylab import *
import pandas as pd
import numpy as np
import os
import matplotlib



from scipy import stats, integrate

import matplotlib.pyplot as plt









lapd_data = pd.read_csv('/home/sam/Downloads/Arrest_Data_from_2010_to_Present.csv')
nsduh_data = pd.read_csv('/home/sam/Downloads/35509-0001-Data.tsv', sep = '\t')
nsduh_addy = nsduh_data[['NEWRACE2','ADDERALL']]
lapd_charge = lapd_data[['Charge','Descent Code']]







