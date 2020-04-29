#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn


# In[2]:


df=pd.read_csv('housing_in_london.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[7]:


df.drop(['no_of_houses','date'],axis=1,inplace=True)
df.drop(['recycling_pct','life_satisfaction','median_salary','mean_salary'],axis=1,inplace=True)


# In[8]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[9]:


df['no_of_crimes']=df['no_of_crimes'].fillna(df['no_of_crimes'].mean())


# In[10]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[11]:


df['houses_sold']=df['houses_sold'].fillna(df['houses_sold'].mean())
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[12]:


df.info()


# In[13]:


df.shape


# In[14]:


df['code']=df.code.str.replace('E','').astype(float)


# In[15]:


df.info()


# In[16]:


df['area'] = pd.factorize(df.area)[0]
df['area'] = df['area'].astype("float")


# In[17]:


df.info()


# In[18]:


df.head()


# In[19]:


df.describe()


# In[39]:


X=df[['area','code','houses_sold','no_of_crimes','borough_flag']]
y=df[['average_price']]


# In[33]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=1,)


# In[34]:


from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor(random_state=0,)


# In[35]:


model.fit(X_train,y_train)


# In[36]:


prediction=(model.predict(X_test).astype(int))


# In[37]:


print("predictions:",prediction)


# In[ ]:


from sklearn.metrics import r2_score
r2_score(prediction,y_test)


# In[ ]:





# In[ ]:





# In[ ]:




