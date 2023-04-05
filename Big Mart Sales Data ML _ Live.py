#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
style.use('ggplot')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score


# In[2]:


df = pd.read_csv('BigMart Sales Data.csv')
df.head()


# In[3]:


df.info()


# In[4]:


categorical_values = df.select_dtypes(include=[object])
print("Count of categorical features in the dataset :",categorical_values.shape[1])

numerical_values = df.select_dtypes(include=[np.float64, np.int64])
print("Count of Nummerial features in the dataset : ",numerical_values.shape[1])


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


sns.boxplot(x = df['Item_Weight'])


# In[8]:


df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())


# In[9]:


sns.countplot(x='Outlet_Size', data=df)


# In[10]:


df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])


# In[11]:


df.isnull().sum()


# In[12]:


df['Item_Fat_Content'].value_counts()


# In[13]:


df.replace({'Item_Fat_Content':{'low fat':'Low Fat','LF':'Low Fat','reg':'Regular'}}, inplace=True)


# In[14]:


df['Item_Fat_Content'].value_counts()


# In[15]:


df['Item_Type'].value_counts()


# In[16]:


df['Outlet_Location_Type'].value_counts()


# In[17]:


df['Outlet_Type'].value_counts()


# In[18]:


encoder = LabelEncoder()


# In[19]:


df['Item_Identifier'] = encoder.fit_transform(df['Item_Identifier'])
df['Item_Fat_Content'] = encoder.fit_transform(df['Item_Fat_Content'])
df['Item_Type'] = encoder.fit_transform(df['Item_Type'])
df['Outlet_Identifier'] = encoder.fit_transform(df['Outlet_Identifier'])
df['Outlet_Size'] = encoder.fit_transform(df['Outlet_Size'])
df['Outlet_Location_Type'] = encoder.fit_transform(df['Outlet_Location_Type'])
df['Outlet_Type'] = encoder.fit_transform(df['Outlet_Type'])


# In[20]:


df.info()


# In[21]:


df.head()


# In[22]:


X = df.drop(columns='Item_Outlet_Sales', axis=1)
Y = df['Item_Outlet_Sales']


# In[23]:


print("X --> ",X.shape)
print("Y --> ", Y.shape)


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=42)


# In[25]:


print("X_train: ",X_train.shape)
print("X_test: ",X_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)


# In[26]:


linreg = LinearRegression()
linreg.fit(X_train, y_train)
linreg_pred = linreg.predict(X_test)


# In[27]:


linreg_mae = mean_absolute_error(y_test, linreg_pred)
linreg_r2 = r2_score(y_test, linreg_pred)
print("MAE of the linear regression model is: ",linreg_mae)
print("R2 score of the linear regression model is: ",linreg_r2)


# In[28]:


xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)


# In[29]:


xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)
print("MAE of the XGBoost model is: ",xgb_mae)
print("R2 score of the XGBoost model is: ",xgb_r2)


# In[31]:


rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


# In[32]:


rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print("MAE of the Random forest model is: ",rf_mae)
print("R2 score of the Random forest model is: ",rf_r2)


# In[33]:


rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=100, n_jobs=5, 
                           random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


# In[34]:


rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print("MAE of the Random forest model is: ",rf_mae)
print("R2 score of the Random forest model is: ",rf_r2)


# In[35]:


df.head(1)


# In[36]:


df.columns


# In[38]:


data = {'Item_Identifier':150, 'Item_Weight':10, 'Item_Fat_Content':0, 'Item_Visibility':0.015,
       'Item_Type':4, 'Item_MRP':250, 'Outlet_Identifier':9,
       'Outlet_Establishment_Year':2000, 'Outlet_Size':1, 'Outlet_Location_Type':0,
       'Outlet_Type':1}
index= [0]
new_df = pd.DataFrame(data, index)
new_df


# In[39]:


value_pred = xgb.predict(new_df)
print("The outlet sales value for the new data is: ",value_pred)


# In[ ]:




