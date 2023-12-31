#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os #paths to file
import numpy as np # linear algebra
import pandas as pd # data processing
import warnings# warning filter


#ploting libraries
import matplotlib.pyplot as plt 
import seaborn as sns

#feature engineering
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#train test split
from sklearn.model_selection import train_test_split

#metrics
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.model_selection  import cross_val_score as CVS


#ML models
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso


#default theme and settings
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)
pd.options.display.max_columns

#warning hadle
warnings.filterwarnings("always")
warnings.filterwarnings("ignore")


# In[2]:


data_train= pd.read_csv(r"C:\Users\hegde\Desktop\DS Project\Train.csv")
data_train.head()


# In[3]:


data_test= pd.read_csv(r"C:\Users\hegde\Desktop\DS Project\Test.csv")
data_test.head()


# In[4]:


data_train.shape


# In[5]:


data_train.isnull().sum()


# In[6]:


data_test.isnull().sum()


# In[7]:


data_train.isnull().sum()/data_train.shape[0]*100


# In[8]:


data_train['Item_Weight']


# In[9]:


sns.boxplot(data = data_train['Item_Weight'], orient = 'v', color = 'r')
plt.title("Item_Weight_Boxplot")


# In[10]:


data_train['Item_Weight'].fillna(data_train['Item_Weight'].mean(),inplace=True)   #Train
data_test['Item_Weight'].fillna(data_test['Item_Weight'].mean(),inplace=True)     #Test


# In[11]:


data_train['Item_Weight'].isnull().sum(), data_test['Item_Weight'].isnull().sum()


# In[12]:


data_train['Outlet_Size'].value_counts()


# In[13]:


data_train['Outlet_Size'].mode()
data_test['Outlet_Size'].mode()


# In[14]:


data_train['Outlet_Size'].fillna(data_train['Outlet_Size'].mode()[0],inplace=True)
data_test['Outlet_Size'].fillna(data_test['Outlet_Size'].mode()[0],inplace=True)


# In[15]:


data_train.isnull().sum()


# In[16]:


data_test.isnull().sum()


# In[17]:


data_train['Item_Fat_Content'].value_counts()


# In[18]:


data_train.replace({'Item_Fat_Content':{'Low Fat': 'LF', 'low fat':'LF','reg':'Regular'}},inplace=True)
data_test.replace({'Item_Fat_Content':{'Low Fat': 'LF', 'low fat':'LF','reg':'Regular'}},inplace=True)


# In[19]:


data_train['Item_Fat_Content'].value_counts()  


# In[20]:


data_test['Item_Fat_Content'].value_counts()  


# In[21]:


data_train.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)
data_test.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)


# In[22]:


data_train.head()


# In[23]:


data_train.duplicated().any()


# In[24]:


data_test.duplicated().any()


# In[25]:


plt.figure(figsize=(6,4))
sns.countplot(x='Item_Fat_Content' , data=data_train ,palette='rocket')
plt.xlabel('Item_Fat_Content', fontsize=14)
plt.show()


# In[26]:


plt.figure(figsize=(27,10))
sns.countplot(x='Item_Type' , data=data_train ,palette='rocket')
plt.xlabel('Item_Type', fontsize=14)
plt.show()


# In[27]:


plt.figure(figsize=(10,4))
sns.countplot(x='Outlet_Size' , data=data_train ,palette='rocket')
plt.xlabel('Outlet_Size', fontsize=14)
plt.show()


# In[28]:


plt.figure(figsize=(10,4))
sns.countplot(x='Outlet_Location_Type' , data=data_train ,palette='rocket')
plt.xlabel('Outlet_Location_Type', fontsize=14)
plt.show()


# In[29]:


plt.figure(figsize=(10,4))
sns.countplot(x='Outlet_Type' , data=data_train ,palette='rocket')
plt.xlabel('Outlet_Type', fontsize=14)
plt.show()


# In[30]:


plt.hist(data_train['Outlet_Establishment_Year'])
plt.title("Outlet_Establishment_Year")
plt.show()


# In[31]:


num = data_train.select_dtypes('number').columns.to_list()
data_num =  data_train[num]


# In[32]:


for numeric in data_num[num[:3]]:
    plt.scatter(data_num[numeric], data_num['Item_Outlet_Sales'])
    plt.title(numeric)
    plt.ylabel('Item_Outlet_Sales')
    plt.show()


# In[33]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[34]:


data_train['Item_Fat_Content']= le.fit_transform(data_train['Item_Fat_Content'])
data_train['Item_Type']= le.fit_transform(data_train['Item_Type'])
data_train['Outlet_Size']= le.fit_transform(data_train['Outlet_Size'])
data_train['Outlet_Location_Type']= le.fit_transform(data_train['Outlet_Location_Type'])
data_train['Outlet_Type']= le.fit_transform(data_train['Outlet_Type'])


# In[35]:


data_train.head()


# In[36]:


X=data_train.drop('Item_Outlet_Sales',axis=1)


# In[37]:


Y=data_train['Item_Outlet_Sales']


# In[38]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=101, test_size=0.2)


# In[39]:


X.describe()


# In[40]:


from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
sc= StandardScaler()


# In[41]:


X_train_std= sc.fit_transform(X_train)


# In[42]:


X_test_std= sc.transform(X_test)


# In[43]:


X_train_std


# In[44]:


X_test_std


# In[45]:


Y_train


# In[46]:


Y_test


# In[93]:


X.head()


# In[94]:
import joblib

joblib.dump(sc,'sc.sav')


# In[47]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[48]:


from sklearn.linear_model import LinearRegression
lr= LinearRegression()


# In[49]:


lr.fit(X_train_std,Y_train)


# In[50]:


y_pred_lr=lr.predict(X_test_std)


# In[51]:


train_score_ = lr.score(X_train,Y_train)
train_score_


# In[52]:


r2_score(Y_test,y_pred_lr)


# In[53]:


print(f"R2_score:")
print(r2_score(Y_test,y_pred_lr))
print(f"MAE:")
print(mean_absolute_error(Y_test,y_pred_lr))
print(f"MSE:")
print(np.sqrt(mean_squared_error(Y_test,y_pred_lr)))


# In[54]:


rmse_ll=np.sqrt(mean_squared_error(Y_test,y_pred_lr))
rmse_ll


# In[55]:


#cross val score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[56]:


kf2 = KFold(n_splits=5, shuffle=True, random_state=42)


# In[57]:


cv_score_ = cross_val_score(lr, X, Y, cv=kf2, scoring='r2')
print(f"Cross Val Score:{cv_score_}")
print(f'Mean R-squared: {np.mean(cv_score_)}')


# In[58]:


print("Adjusted R2_score:")
print(1-(1-lr.score(X_train_std,Y_train))*(len(Y_train)-1)/(len(Y_train)-X_train_std.shape[1]-1))


# In[59]:


from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor()


# In[60]:


rf.fit(X_train_std,Y_train)


# In[61]:


RF_train_score= rf.score(X_train,Y_train)
RF_train_score


# In[62]:


y_pred_rf= rf.predict(X_test_std)


# In[63]:


r2_score(Y_test,y_pred_rf)


# In[64]:


print(f"R2_score:")
print(r2_score(Y_test,y_pred_rf))
print(f"MAE:")
print(mean_absolute_error(Y_test,y_pred_rf))
print(f"MSE:")
print(np.sqrt(mean_squared_error(Y_test,y_pred_rf)))


# In[65]:


rmse=np.sqrt(mean_squared_error(Y_test,y_pred_rf))
rmse


# In[66]:


print("Adjusted R2_score:")
print(1-(1-rf.score(X_train_std,Y_train))*(len(Y_train)-1)/(len(Y_train)-X_train_std.shape[1]-1))


# In[67]:


kf2 = KFold(n_splits=5, shuffle=True, random_state=42)


# In[68]:


cv_score_1 = cross_val_score(rf, X, Y, cv=kf2, scoring='r2')
print(f"Cross Val Score:{cv_score_1}")
print(f'Mean R-squared: {np.mean(cv_score_1)}')


# In[69]:


from xgboost import XGBRegressor
xg= XGBRegressor()


# In[70]:


xg.fit(X_train_std, Y_train)


# In[71]:


y_pred_xg= xg.predict(X_test_std)


# In[72]:


train_score_ = xg.score(X_train,Y_train)
train_score_


# In[73]:


r2_score(Y_test,y_pred_xg)


# In[74]:


print(f"R2_score:")
print(r2_score(Y_test,y_pred_xg))
print(f"MAE:")
print(mean_absolute_error(Y_test,y_pred_xg))
print(f"MSE:")
print(np.sqrt(mean_squared_error(Y_test,y_pred_xg)))


# In[75]:


print("Adjusted R2_score:")
print(1-(1-xg.score(X_train_std,Y_train))*(len(Y_train)-1)/(len(Y_train)-X_train_std.shape[1]-1))


# In[76]:


# Creating KFold object
KF = KFold(n_splits=5, shuffle=True, random_state=42)


# In[77]:


cv_scores = cross_val_score(xg, X, Y, cv=KF, scoring='neg_mean_squared_error')


# In[78]:


# Converting negative MSE to positive RMSE
cv_rmse_scores = np.sqrt(-cv_scores)


# In[79]:


print(f"CV RMSE Scores: {cv_rmse_scores}")
print(f"Mean RMSE: {np.mean(cv_rmse_scores)}")


# In[80]:


KF1 = KFold(n_splits=6, shuffle=True, random_state=42)


# In[81]:


cv_scores_r2 = cross_val_score(xg, X, Y, cv=KF1, scoring='r2')


# In[82]:


print(f'Cross-Validation R-squared Scores: {cv_scores_r2}')
print(f'Mean R-squared: {np.mean(cv_scores_r2)}')


# In[83]:


RandomForest_data = {
    'Model': ['Random Forest'],
    'MAE':[805.06],
    'MSE': [1157.65],
    'RMSE': [1158.41],
    'R2 Score': [0.54],
    'Adjusted R2 Score': [0.93],
    'Cross-Validation Score': [0.54]
}

Linear_Regression= {
    'Model': ['Linear Regression'],
    'MAE':[837.74],
    'MSE': [1133.81],
    'RMSE': [1133.81],
    'R2 Score': [0.55],
    'Adjusted R2 Score': [0.55],
    'Cross-Validation Score': [0.54]
}


XGBOOST_data = {
    'Model': ['XGBoost'],
    'MAE':[881.71],
    'MSE': [1249.82],
    'R2 Score': [0.46],
    'Adjusted R2 Score': [0.97],
    'Cross-Validation Score': [0.53],
}


# In[84]:


RANDOMFOREST_df = pd.DataFrame(RandomForest_data)
Linear_Regression_df = pd.DataFrame(Linear_Regression)
XGBOOST_df = pd.DataFrame(XGBOOST_data)
# Concatenate DataFrames
summary_table = pd.concat([RANDOMFOREST_df, Linear_Regression_df, XGBOOST_df], ignore_index=True)


# In[85]:


summary_table


# In[88]:


import joblib


# In[89]:


joblib.dump(lr,'lr.sav')


# In[90]:


loaded_model = joblib.load('lr.sav')


# In[91]:


loaded_model.predict(X_test_std)


# In[ ]:




