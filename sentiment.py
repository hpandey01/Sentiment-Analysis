
# coding: utf-8

# In[130]:


import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, cross_val_score


# In[131]:


# input file for training
excel_file = 'Languages_Sentiment_labels.xlsx'
lang= pd.read_excel(excel_file)


# In[132]:


# taking training data into list
sentiment=lang['Sentiment'].tolist()
language=lang['language'].tolist()
text=lang['text'].tolist()


# In[133]:


# reading the test file
name=input("Enter file name: ")
excel_file1=name
lang1= pd.read_excel(excel_file1)


# In[134]:


# taking test data into list
sentiment1=lang['Sentiment'].tolist()
language1=lang['language'].tolist()
text1=lang['text'].tolist()


# In[135]:


# code for data pre-processing
def clean(text):
    text_cln=[re.sub('@[A-Za-z0-9]+','',i) for i in text]
    text_cln=[re.sub('#[A-Za-z0-9]+','',i) for i in text_cln]
    text_cln=[re.sub('https[A-Za-z0-9./]+','',i) for i in text_cln]
    text_cln=[re.sub(r'\S*\d\S*','',i) for i in text_cln]
    text_cln=[re.compile("(\.)|(\;)|(\-)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\_)(\||)").sub("",i.lower()) for i in text_cln]
    text_cln=[re.sub(r'\r\n',' ',i) for i in text_cln]
    emoji_pattern = re.compile("["
            u"\U0001F600-\U000FFFFF"
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text_cln=[emoji_pattern.sub(r'',i) for i in text_cln] # no emoji
    text_cln=[i.replace(u"\ufffd", "") for i in text_cln]
    text_cln
    text_cln1=[]
    for i in text_cln:
        spl=i.split()
        for j in spl:
            if j=='rt':
                spl.remove(j)
            if j=='&amp':
                spl.remove(j)
            if j[0:4]=="http":
                spl.remove(j)
            if j[-3:-1]=="...":
                spl.remove(j)
        spl1=' '.join(spl)
        text_cln1+=[spl1]
    text_cln=text_cln1
    text_cln=[re.sub('[!$@:)#.;,?&|"\']', '',i) for i in text_cln]
    
    return(text_cln)


# In[136]:


# one hot encoding of training and testing data
def encoding(text_cln,text_cln1):
    cv = CountVectorizer(binary=True,ngram_range=(1, 3))
    cv.fit(text_cln+text_cln1)
    x= cv.transform(text_cln)
    x1=cv.transform(text_cln1)
    x=[x,x1]
    return(x)


# In[137]:


# giving train sentiment a numerical value
sen=[]
for i in sentiment:
    if i=="neutral":
        sen+=[0]
    if i=="positive":
        sen+=[1]
    if i=="negative":
        sen+=[-1]


# In[138]:


# giving test sentiment a numerical value
sen1=[]
for i in sentiment1:
    if i=="neutral":
        sen1+=[0]
    if i=="positive":
        sen1+=[1]
    if i=="negative":
        sen1+=[-1]


# In[139]:


# dividing the text and sentiment according to the language
lan=['bn','mr','en','hi','ta','te']

text_cln=clean(text)                  # for training data
y=[[],[],[],[],[],[]]            
z=[[],[],[],[],[],[]]
for i in range(0,len(language)):
    for j in range(0,len(lan)):
        if language[i]==lan[j]:
            y[j]+=[text_cln[i]]
            z[j]+=[sen[i]]
            
text_cln1=clean(text1)                  # for testing data
y1=[[],[],[],[],[],[]]
z1=[[],[],[],[],[],[]]
for i in range(0,len(language1)):
    for j in range(0,len(lan)):
        if language1[i]==lan[j]:
            y1[j]+=[text_cln1[i]]
            z1[j]+=[sen1[i]]
            
x=[]                                 # vectors of training and testing data
x1=[]
for i in range(0,len(lan)):
    var=encoding(y[i],y1[i])
    x+=[var[0]]
    x1+=[var[1]]


# In[152]:


# logistic regression classifier for each language
pr=[[],[],[],[],[],[]]
for i in range(0,6):
    if i==0:
        lr0 = LogisticRegression(solver='saga')
        lr0.fit(x[i],z[i])
        pr[i]=lr0.predict(x1[i])
    if i==1:
        lr1 = LogisticRegression(solver='saga')
        lr1.fit(x[i],z[i])
        pr[i]=lr1.predict(x1[i])
    if i==2:
        lr2 = LogisticRegression(solver='saga')
        lr2.fit(x[i],z[i])
        pr[i]=lr2.predict(x1[i])
    if i==3:
        lr3 = LogisticRegression(solver='saga')
        lr3.fit(x[i],z[i])
        pr[i]=lr3.predict(x1[i])
    if i==4:
        lr4 = LogisticRegression(solver='saga')
        lr4.fit(x[i],z[i])
        pr[i]=lr4.predict(x1[i])
    if i==5:
        lr5 = LogisticRegression(solver='saga')
        lr5.fit(x[i],z[i])
        pr[i]=lr5.predict(x1[i])


# In[166]:


# giving score to each sentiment
# by multiplying each vector to the corresponding coffecients obtained from logistic regression classifier
xarr=[[],[],[],[],[],[]]
score=[[],[],[],[],[],[]]
for i in range(0,6):
    xarr[i]=x1[i].toarray()
    if i==0:
        clas=lr0
    if i==1:
        clas=lr1
    if i==2:
        clas=lr2
    if i==3:
        clas=lr3
    if i==4:
        clas=lr4
    if i==5:
        clas=lr5
    
    for l in range(0,len(xarr[i])):
        num=0
        for j in range(0,len(clas.coef_[0])):
            num+=clas.coef_[(z1[i][l]+1)][j]*xarr[i][l][j]
        score[i]+=[num]
maxi=[[],[],[],[],[],[]]
for i in range(0,6):
    maxi[i]=max(score[i])
max_score=max(maxi)
for i in range(0,6):
    for j in range(0,len(score[i])):
        score[i][j]/=max_score
print(pr)
print(score)

