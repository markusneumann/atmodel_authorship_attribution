
# coding: utf-8

# # Authorship detection with SVM
# 
# Load the data:

# In[1]:

import pandas as pd
df = pd.read_csv('data/reddit2010-06_subset.csv')

#rename columns
df = df.rename(columns={'author': 'y', 'body': 'X'})

#get rid of subreddit column as well
df = df[['y', 'X']]


# In[2]:

df.head()


# In[3]:

#test-training split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['X'], df['y'], test_size=0.1, random_state=42)


# ScikitLearn pipeline to transform the text to a document-term matrix, do tf-idf transformation, and then apply SVM with stochastic gradient descent.

# In[4]:

#Pre-processing and SVM pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=0.00005, random_state=42)),
])


# Fit the model and do prediction.

# In[6]:

#get_ipython().run_cell_magic('time', '', '#fit and predict\ntext_clf.fit(X_train, y_train)\npredicted = text_clf.predict(X_test)')

#fit and predict
text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)

# Assess accuracy - about 15%, not great.

# In[10]:

#accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, predicted))
#print out the accuracy for every single author, if desired
#print(metrics.classification_report(y_test, predicted))


# Do a grid search to search for the optimal alpha (regularization) parameter. Doesn't seem to help much.

# In[11]:

#Grid search
from sklearn.model_selection import GridSearchCV
parameters = {'clf__alpha': (0.00005, 0.0005, 0.005, 0.05, 0.5, 1),
}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, y_train)
gs_clf.best_score_


# In[12]:

#best parameters from grid search
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

