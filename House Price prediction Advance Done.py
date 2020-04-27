#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('train.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.isnull().sum()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[5]:


df.info()


# In[6]:


df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
df.drop(['Alley'],axis=1,inplace=True)
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df.drop(['GarageYrBlt'],axis=1,inplace=True)
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[7]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[8]:


df.drop(['Id'],axis=1,inplace=True)


# In[9]:


df.shape


# In[10]:


df.isnull().sum()


# In[11]:


df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])


# In[12]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[13]:


df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])


# In[14]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[15]:


df.shape


# In[16]:


df.head()


# In[17]:


df.select_dtypes(include=['object']).columns


# In[18]:


columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']


# In[19]:


len(columns)


# In[20]:


#converting df into categorical features

def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final


# In[21]:


main_df=df.copy()


# In[22]:


test_df=pd.read_csv('formulatedtest.csv')


# In[23]:


test_df.shape


# In[24]:


test_df.head()


# In[25]:


final_df=pd.concat([df,test_df],axis=0)


# In[ ]:





# In[26]:


final_df.shape


# In[27]:


final_df=category_onehot_multcols(columns)


# In[28]:


final_df.shape


# In[29]:


final_df =final_df.loc[:,~final_df.columns.duplicated()]


# In[30]:


final_df.shape


# In[31]:


final_df.head()


# In[32]:


df_Train=final_df.iloc[:1460,:]
df_Test=final_df.iloc[1460:,:]


# In[33]:


df_Test.drop(['SalePrice'],axis=1,inplace=True)


# In[34]:


df_Test.shape


# In[35]:


X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']


# In[36]:


import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics


# In[37]:


model=DecisionTreeRegressor()


# In[38]:


model.fit(X_train,y_train)


# In[39]:


y_pred=model.predict(df_Test)


# In[40]:


y_pred


# In[41]:


pred=pd.DataFrame(y_pred)


# In[42]:


sub_df=pd.read_csv('sample_submission.csv')


# In[43]:


dataset=pd.concat([sub_df['Id'],pred],axis=1)
dataset.columns=['Id','SalePrice']


# In[44]:


dataset.to_csv('sample_submission.csv',index=False)


# In[ ]:





# In[ ]:




