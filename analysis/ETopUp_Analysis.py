# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 17:07:52 2017

@author: Rigved
"""

#data analysis and wrangling
import pandas as pd
#import numpy as np
#import random as rnd

# visualization
import matplotlib.pyplot as plt
#import plotly.offline as py  
#import plotly.graph_objs as go
import seaborn as sns

train_df = pd.read_csv('test2.csv',parse_dates=[[4,5]],error_bad_lines=False)#[['Created on', 'Time']]

train_df.head()
train_df.tail()

train_df.info()
train_df.describe()

train_df.describe(include=['O'])

#train_df[['Net_price', 'Created by']].groupby(['Created by'], as_index=False).mean().sort_values(by='Net_price', ascending=False)
#train_df[['Net price', 'Created by']].groupby(['Created by'], as_index=False).sum().sort_values(by='Net price', ascending=False)
sg = train_df[['Net_price', 'SG']].groupby(['SG'], as_index=False).sum().sort_values(by='Net_price', ascending=False)
sgsup = train_df[['Net_price', 'SGsup']].groupby(['SGsup'], as_index=False).sum().sort_values(by='Net_price', ascending=False)

df = train_df[['Net price', 'State']].groupby(['State'], as_index=False).sum().sort_values(by='Net price', ascending=False)

g = sns.FacetGrid(df, col='State')
g.map(plt.hist, 'Net price')

sns.stripplot(x="State", y="Net price", data=df);
sns.stripplot(x="SG", y="Net price", data=sg);
sns.stripplot(x="SGsup", y="Net price", data=sgsup);
'''py.init_notebook_mode()
Df1 = [go.Scatter(x=train_df.Time, y=train_df.Net_price)]
py.iplot(Df1)'''

df = df.to_string()
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
df = sc_X.fit_transform(df)

df.describe()
df.info()


plt.plot(df.State, df[['Net price']])
plt.show()

df.describe()