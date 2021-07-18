#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pds
import numpy as npy
import seaborn as sb
import warnings
warnings.filterwarnings('ignore')


# In[4]:


data = pds.read_csv('D:\password-strength\data.csv', error_bad_lines = False)
data.head()


# In[6]:


data['strength'].unique()


# In[ ]:


# 0 - poor
# 2 - strong
# 1 - normal


# In[7]:


data.isna().sum()   #how many has NaN value


# In[8]:


data[data['password'].isnull()]    # what's inside first '[]' is the filter


# In[9]:


# drop it
data.dropna(inplace = True)


# In[10]:


data.isnull().sum()


# In[11]:


sb.countplot(data['strength'])


# In[ ]:


# 1 has the highest count, imbalanced dataset     
#strength is a dependent feature. Separate them


# In[12]:


password_tuple = npy.array(data)


# In[13]:


password_tuple


# In[ ]:


# shuffle the data so it provides robustness to the model


# In[14]:


import random
random.shuffle(password_tuple)


# In[15]:


#separation
x = [labels[0] for labels in password_tuple]
y = [labels[1] for labels in password_tuple]


# In[16]:


x    #all passwords


# In[ ]:


# TF-IDF


# In[28]:


def word_divide_char(inputs):
    character = []                  
    for i in inputs:                # password data gets split into single chars
        character.append(i)
    return character


# In[29]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[30]:


vectorizer = TfidfVectorizer(tokenizer = word_divide_char)


# In[31]:


X = vectorizer.fit_transform(x)


# In[32]:


X.shape


# In[ ]:


# number of cols increased as now it is in the form of a vector.


# In[33]:


vectorizer.get_feature_names()


# In[34]:


first_doc_vector = X[0]
first_doc_vector


# In[35]:


first_doc_vector.T.todense()


# In[ ]:


#prepare data for model


# In[38]:


df = pds.DataFrame(first_doc_vector.T.todense(), index = vectorizer.get_feature_names(), columns = ['TF-IDF'])
df.sort_values(by = ['TF-IDF'], ascending = False)


# In[ ]:


# pass data


# In[ ]:


# apply logistic Regression (ML Algo)


# In[39]:


from sklearn.model_selection import train_test_split


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)  #returns 4 param
# test size - here-20% of data is for testing and 80% for training.


# In[42]:


X_train.shape


# In[ ]:


# 53571 is somewhere close to 80 percent.


# In[43]:


from sklearn.linear_model import LogisticRegression


# In[44]:


clf = LogisticRegression(random_state = 0, multi_class = 'multinomial')
# multinomial class is considered because we have 3 categories (0, 1, 2) 


# In[45]:


clf.fit(X_train, y_train)


# In[ ]:


# now it's time for predictions....!


# In[46]:


dt = npy.array(['@#123abcd'])
predc = vectorizer.transform(dt)
clf.predict(predc)


# In[ ]:





# In[47]:


y_predc = clf.predict(X_test)
y_predc      # all predictions in the form of array.


# In[ ]:





# In[48]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[49]:


cm = confusion_matrix(y_test, y_predc)
print(cm)
print(accuracy_score(y_test, y_predc))


# In[ ]:


# 5271 92798 11529 are true predictions, 
   #0.8183352248969595 shows that model has an accuracy of approx 82 percent


# In[50]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_predc))


# In[ ]:




