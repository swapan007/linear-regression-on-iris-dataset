#!/usr/bin/env python
# coding: utf-8

# # EXPLORING IRIS DATASET

# In[25]:


# IMPORTING THE REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[26]:


# importing dataset
iris=pd.read_csv("Iris.csv")


# In[27]:


iris.shape


# In[28]:


iris.head()


# In[29]:


iris.columns=["id","sepal_length","sepal_width","petal_length","petal_width","species"]


# In[31]:


iris.head()


# In[32]:


iris.tail()


# In[34]:


# sepal width >4
iris[iris["sepal_width"]>4]


# # LINEAR REGRESSION MODEL1 [USING 1 FEATURE]

# In[39]:


#breaking into training and testing dataset
y=iris[["sepal_length"]]
x=iris[["sepal_width"]]


# In[44]:


from sklearn.model_selection import train_test_split


# In[43]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[47]:


x_test.shape


# In[49]:


from sklearn.linear_model import LinearRegression


# In[50]:


lr=LinearRegression()


# In[51]:


lr.fit(x_train,y_train)


# In[52]:


y_pred=lr.predict(x_test)


# In[54]:


y_test.head()


# In[57]:


y_pred[:5]


# In[62]:


# finding error
from sklearn.metrics import mean_squared_error as mse
mse(y_test,y_pred)


# ## MODEL 2 USING MULTIPLE FEATURES 

# In[63]:


# BREAKING INTO TRAINING AND TESTING DATASET
x=iris[["sepal_width","petal_length","petal_width"]]
y=iris[["sepal_length"]]


# In[64]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[66]:


lr.fit(x_train,y_train)


# In[69]:


y_pred=lr.predict(x_test)
y_pred[:5]


# In[70]:


y_test.head()


# In[73]:


mse(y_test,y_pred)


# In[ ]:


# as we can see error of model 2 is very less as compared to model 1 as we take multiple independent variables in model 2

