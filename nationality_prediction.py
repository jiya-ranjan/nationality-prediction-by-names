#!/usr/bin/env python
# coding: utf-8

# predicting nationality by names

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


# In[2]:


df = pd.read_csv("names.csv")


# In[3]:


df.head()


# In[4]:


df.groupby('Country')['Name'].size()


# In[5]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Country'])


# In[6]:


y


# In[7]:


x_train, x_test, y_train, y_test = train_test_split(df['Name'], y, test_size=0.2,random_state=0)
vectorizer = CountVectorizer().fit(x_train)


# In[8]:


transformed_x_train = vectorizer.transform(x_train)


# In[9]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(transformed_x_train, y_train)


# In[10]:


def predict(names, label_str=False):
        name_vector = vectorizer.transform(names)
        pred = clf.predict(name_vector)
        if not label_str:
            return pred
        else:
            return label_encoder.inverse_transform(pred.ravel()).ravel()
        
        
def evaluate(labels, prediction):
        cm = confusion_matrix(labels, prediction)
        # recall
        recall = np.diag(cm) / np.sum(cm, axis = 1)
        # precision
        precision = np.diag(cm) / np.sum(cm, axis = 0)

        acc = (prediction == labels).mean()

        return {'accuracy':acc, 'precision':precision, 'recall':recall}

def plot_confusion( yt, prediction_test):
        cm = confusion_matrix(yt, prediction_test)
        fig = plt.figure(figsize=(15, 10))
        plt.imshow(cm, interpolation='nearest')
        plt.colorbar()
        axis_font = {'size': 13, 'color':'black'}
        cat = label_encoder.classes_
        num_class = len(cat)
        classNames = [cat[i] for i in range(num_class)]
        plt.title("Confusion Matrix by class", fontdict=axis_font)
        plt.ylabel("True Label", fontdict=axis_font)
        plt.xlabel("Predicted Label", fontdict=axis_font)
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        fdic = {'size':10, 'color':'white', 'weight':'heavy'}
        for i in range(num_class):
            for j in range(num_class):
                plt.text(j, i, str(cm[i, j]), fontdict=fdic, horizontalalignment='center',verticalalignment='center')
        plt.show()


# In[11]:


print(predict(["ram"],True))


# In[12]:


y_pred = predict(x_test)


# In[13]:


evaluate(y_pred,y_test)


# In[14]:


plot_confusion(y_pred,y_test)



import joblib

# Save the model
joblib.dump(clf, 'nationality_predictor_model.pkl')

# Save the CountVectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')

# Save the LabelEncoder
joblib.dump(label_encoder, 'label_encoder.pkl')






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




