#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns  
import time 

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from nltk.tokenize import RegexpTokenizer  
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.pipeline import make_pipeline

from PIL import Image

import pickle 


# In[2]:


df= pd.read_csv("C:/Shreya/AIURL/AIURL/Src/data/phishing_site_urls.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


sns.countplot(x="Label",data=df)


# In[7]:


tokenizer = RegexpTokenizer(r'[A-Za-z]+')


# In[12]:


tokenizer.tokenize(df.URL[0])


# In[8]:


print('Getting words tokenized ...')
t0= time.perf_counter()
df['text_tokenized'] = df.URL.map(lambda t: tokenizer.tokenize(t))
t1 = time.perf_counter() - t0
print('Time taken',t1 ,'sec')


# In[9]:


df.sample(10)


# In[10]:


stemmer = SnowballStemmer("english")


# In[11]:


print('Getting words stemmed ...')
t0= time.perf_counter()
df['text_stemmed'] = df['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])
t1= time.perf_counter() - t0
print('Time taken',t1 ,'sec')


# In[12]:


df.sample(10)


# In[13]:


print('Get joiningwords ...')
t0= time.perf_counter()
df['text_sent'] = df['text_stemmed'].map(lambda l: ' '.join(l))
t1= time.perf_counter() - t0
print('Time taken',t1 ,'sec')


# In[14]:


bad_sites = df[df.Label == 'bad']
good_sites = df[df.Label == 'good']


# In[15]:


bad_sites.head()


# In[16]:


good_sites.head()


# In[17]:


df.head()


# In[18]:


cv = CountVectorizer()
feature = cv.fit_transform(df.text_sent) 
feature[:5].toarray() 


# In[19]:


from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(feature, df.Label)


# In[20]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(trainX,trainY)


# In[21]:


lr.score(testX,testY)


# In[22]:


Scores_ml = {}
Scores_ml['Logistic Regression'] = np.round(lr.score(testX,testY),2)


# In[23]:


print('Training Accuracy :',lr.score(trainX,trainY))
print('Testing Accuracy :',lr.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(lr.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(lr.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# In[24]:


from sklearn.naive_bayes import MultinomialNB 
mnb = MultinomialNB()
mnb.fit(trainX,trainY)


# In[25]:


mnb.score(testX,testY)


# In[26]:


Scores_ml['MultinomialNB'] = np.round(mnb.score(testX,testY),2)


# In[27]:


print('Training Accuracy :',mnb.score(trainX,trainY))
print('Testing Accuracy :',mnb.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(mnb.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(mnb.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# In[28]:


acc = pd.DataFrame.from_dict(Scores_ml, orient='index', columns=['Accuracy'])

acc.reset_index(inplace=True)
acc.rename(columns={'index': 'Model'}, inplace=True)

sns.set_style('darkgrid')

sns.barplot(data=acc, x='Model', y='Accuracy')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')


plt.show()


# In[29]:


pipeline_ls = make_pipeline(CountVectorizer(tokenizer = RegexpTokenizer(r'[A-Za-z]+').tokenize,stop_words='english'), LogisticRegression())


# In[30]:


trainX, testX, trainY, testY = train_test_split(df.URL, df.Label)
pipeline_ls.fit(trainX,trainY)


# In[31]:


pipeline_ls.score(testX,testY)


# In[32]:


print('Training Accuracy :',pipeline_ls.score(trainX,trainY))
print('Testing Accuracy :',pipeline_ls.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(pipeline_ls.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(pipeline_ls.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# In[33]:


pickle.dump(pipeline_ls,open('phishing.pkl','wb'))
loaded_model = pickle.load(open('phishing.pkl', 'rb'))
result = loaded_model.score(testX,testY)
print(result)


# In[ ]:




