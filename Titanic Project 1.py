#!/usr/bin/env python
# coding: utf-8

# # Titanic Project
# 
# ### Problem Statement:
#  The Titanic Problem is based on the sinking of the ‘Unsinkable’ ship Titanic in early 1912. It gives you information about multiple people like their ages, sexes, sibling counts, embarkment points, and whether or not they survived the disaster. Based on these features, you have to predict if an arbitrary passenger on Titanic would survive the sinking or not. 
# 
# ### Downlaod Files:
# https://github.com/dsrscientist/dataset1/blob/master/titanic_train.csv
# 

# In[1]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[3]:


df=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset1/master/titanic_train.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


plt.figure
sns.heatmap(df.isnull())
plt.show()


# In[9]:


df=df.drop(columns='Cabin', axis=1) # droping the cabin column


# In[10]:


df['Age'].fillna(df['Age'].mean(),inplace=True)


# In[11]:


df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)


# In[12]:


df.isnull().sum()


# In[13]:


# data analysis


# In[14]:


df.describe()


# ## data visualization

# In[15]:


sns.set()


# In[16]:


#making a count plot for "Survived" column
sns.countplot('Survived', data=df)
plt.show()


# In[17]:


df['Survived'].value_counts()


# * Not many passengers survived the accident
# * 0 is for non survived passengers
# * 1 is for survived passengers
# * out of 891 passengers only 342 survived 

# In[18]:


sns.countplot('Sex', data=df)
plt.show()


# In[19]:


sns.countplot('Sex', hue='Survived', data=df)    
plt.show()


# In[20]:


df.groupby(['Sex','Survived'])['Survived'].count()


# * The number of men on ship is more than female
# * But according to data female survived more than men
# * 233 female survived 81 not
# * 109 men survived 468 not

# In[21]:


sns.countplot('Pclass', data=df)
plt.show()


# there are more people in 3 class

# In[22]:


sns.countplot('Pclass', hue='Survived', data=df)    
plt.show()


# In[23]:


df.groupby(['Pclass','Survived'])['Survived'].count()


# * There are 3 class on shep
# * according to data more people survived who live in 1st lass
# * In 1st class 136 survived 80 not
# * In 2nd class 87 survived 97 not
# * In 3rd class 119 survived 372 not

# In[24]:


df['Embarked'].value_counts() # from where they board


# * S = Southampton
# * C = Cherbourg
# * Q = Queenstown

# In[25]:


df.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)


# In[26]:


df.head()


# #### separating features and Target

# In[27]:


x = df.drop(columns=['PassengerId','Name','Ticket','Survived'], axis=1)
y = df['Survived']


# In[28]:


print(x)


# In[29]:


print(y)


# ## Spliting the data

# In[30]:


x_train, x_test, y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=5)


# In[31]:


print(x.shape, x_train.shape, x_test.shape)


# ## Logistic Regression
# 
# Logistic regression is applied to predict the categorical dependent variable. In other words, it's used when the prediction is categorical, for example, yes or no, true or false, 0 or 1. The predicted probability or output of logistic regression can be either one of them, and there's no middle ground.

# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score


# In[33]:


model=LogisticRegression()


# In[34]:


model.fit(x_train,y_train)


# In[35]:


model.score(x_train,y_train)


# In[36]:


model.score(x_test,y_test)


# In[ ]:




