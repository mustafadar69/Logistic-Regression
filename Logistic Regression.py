#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


train=pd.read_csv('titanic_train.csv')


# In[5]:


train.head()


# In[6]:


#checking the null values in the data


# In[10]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[15]:


sns.countplot(x='Survived',data=train)


# In[16]:


sns.countplot(x='Survived',hue='Sex',data=train)


# In[19]:


sns.distplot(train['Age'],kde=False,bins=60)


# In[20]:


train.head()


# In[21]:


#returning the average values of age based on the pclass
def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 27
    
    else:
        return Age
        


# In[22]:


train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)


# In[23]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[24]:


train.drop('Cabin',axis=1,inplace=True)


# In[25]:


train.head()


# In[26]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[27]:


#changing the sex column into the binary column

sex=pd.get_dummies(train['Sex'],drop_first=True)


# In[29]:


sex.head()


# In[31]:


embark=pd.get_dummies(train['Embarked'],drop_first=True)


# In[33]:


embark.head()


# In[35]:


train=pd.concat([train,sex,embark],axis=1)


# In[36]:


train.head()


# In[83]:


train.drop(['Name','Sex','Ticket','Embarked','PassengerId'],axis=1,inplace=True)


# In[43]:


#pclass=pd.get_dummies(train['Pclass'],drop_first=True)
#pclass.head()


# In[84]:


#taking the testing and training values
X=train.drop('Survived',axis=1)
y=train['Survived']
X


# In[85]:


from sklearn.model_selection import train_test_split


# In[86]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# In[87]:


from sklearn.linear_model import LogisticRegression


# In[88]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[90]:


#prediction
predictions=logmodel.predict(X_test)


# In[ ]:


#classification


# In[94]:


from sklearn.metrics import classification_report


# In[95]:


print(classification_report(y_test,predictions))


# In[96]:


from sklearn.metrics import confusion_matrix


# In[97]:


confusion_matrix(y_test,predictions)


# In[ ]:




