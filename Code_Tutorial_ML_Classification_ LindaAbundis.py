#!/usr/bin/env python
# coding: utf-8

# Linda Abundis
# A01636416

#  # 1.1 Read Dataset

# Input(x) -> Comments (review)
# 
# Output(y) -> Sentiments

# In[2]:


import pandas as pd

df_review = pd.read_csv('IMDB Dataset.csv')
df_review


# In[3]:


# 50,000 rows | Balanced data
df_review.value_counts(['sentiment'])


# In[4]:


# Taking 10,000 rows | Unblanced data | Processing will be faster
# 9,000 positives
df_positive = df_review[df_review['sentiment']=='positive'][:9000]
# 1,000 negatives
df_negative = df_review[df_review['sentiment']=='negative'][:1000]

df_review_unb = pd.concat([df_positive, df_negative])
df_review_unb.value_counts(['sentiment'])


# # 1.2 Unbalanced Dataset

# In[5]:


from imblearn.under_sampling import RandomUnderSampler

#Balancing dataset, taking 1,000 from the 9,000 positives (sentiment)
rus = RandomUnderSampler()
df_review_bal, df_review_bal['sentiment']=rus.fit_resample(df_review_unb[['review']], df_review_unb['sentiment'])
#df_review_bal

df_review_bal.value_counts(['sentiment'])



# # 1.3 Dividing data into training and testing

# In[7]:


from sklearn.model_selection import train_test_split

#Specifing data percentage for training and testing
train, test = train_test_split(df_review_bal, test_size=0.33, random_state=42)


# In[8]:


#1340 rows
train


# In[9]:


#660 rows
test


# In[12]:


#Specify x and y variables, input and output
train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']


# # 2 Text Representation (Bag of Words)

# Transform text into numerical data (numerical vectors)
# 
# Two methods: 
# * CountVectorizer:Frequency in which a word appears in a sentence 
# * Tfidf: Relevance that a word has within a sentence, but that it is not so repeated in the other reviews
# 
# 

# # 2.1 Count Vectorizer

# In[13]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
text = ["I love writing code in Python. I love Python code",
        "I hate writing code in Java. I hate Java code"]

df = pd.DataFrame({'review': ['review1', 'review2'], 'text':text})
cv = CountVectorizer(stop_words='english')
cv_matrix = cv.fit_transform(df['text'])
df_dtm = pd.DataFrame(cv_matrix.toarray(), index=df['review'].values, columns=cv.get_feature_names_out())
df_dtm


# # 2.2 Tfidf (term frequency - inverse document frequency)

# In[14]:


#Decimal represents the weight of each word, higher decimal means higher relevance

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
text = ["I love writing code in Python. I love Python code",
        "I hate writing code in Java. I hate Java code"]

df = pd.DataFrame({'review': ['review1', 'review2'], 'text':text})
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text'])
df_dtm = pd.DataFrame(tfidf_matrix.toarray(), index=df['review'].values, columns=tfidf.get_feature_names_out())
df_dtm



# # 2.3 Transforming text data into numerical data (vectors)

# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')

#Fit: finds the best parameters for the function or for the data that we are going to introduce
#Transform: applies the parameters we found to the data
train_x_vector = tfidf.fit_transform(train_x)
test_x_vector = tfidf.transform(test_x)


# 112387 cells will have a value different to cero, because there will be wprds with out many repetitions in many sentences.

# In[16]:


train_x_vector


# Sparse matrix = Matrix with many ceros
# Dense matrix

# # 3 Model Selection
# 

# ML algorithms
# 
# 1. Supervised learning: Input and output are defined.
# 
# * Regression (numerical output)
# * Classification (discrete output)
# ---------------------------------
# 
# * Input: Review
# * Output: Sentiment (discrete)
# 
# 2. Unsupervised learning
# 
# 

# # 3.1 Support Vector Machines (SVM)

# In[17]:


from sklearn.svm import SVC

svc = SVC(kernel='linear')
# Pasing input (numerical) and output (discret)
svc.fit(train_x_vector, train_y)


# # 3.1.1 Testing

# In[18]:


print(svc.predict(tfidf.transform(['A good movie'])))
print(svc.predict(tfidf.transform(['An excellent movie'])))
print(svc.predict(tfidf.transform(['"I did not like this movie at all I gave this movie away"'])))


# # 3.2 Decision Tree

# In[19]:


from sklearn.tree import DecisionTreeClassifier

dec_tree = DecisionTreeClassifier()
# Pasing input (numerical) and output (discret)
dec_tree.fit(train_x_vector, train_y)


# # 3.3 Naive Bayes

# In[20]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
# Pasing input (numerical) and output (discret)
# Pass input to array for the method GaussianNB we are using
gnb.fit(train_x_vector.toarray(), train_y)


# # 3.4 Logistic Regression

# In[21]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_x_vector, train_y)


# # 4 Model Evaluation

# # 4.1 Score (Accuracy)

# Return the mean accuracy on the given test data and labels.

# In[22]:


#Now we will use the testing data
print(svc.score(test_x_vector, test_y))
print(dec_tree.score(test_x_vector, test_y))
print(gnb.score(test_x_vector.toarray(), test_y))
print(lr.score(test_x_vector, test_y))

#In this case SV, is the model with better precision


# # 4.2 F1 Score

# Take into consideration how the data is distributed. If the data is unbalanced, it is convenient to use this model.

# F1 Score is the weighted average of Precision and Recall. Accuracy is used when the True Positives and True negatives are more important while F1-score is used when the False Negatives and False Positives are crucial. Also, F1 takes into account how the data is distributed, so it's useful when you have data with imbalance classes.

# F1 Score = 2(Recall Precision) / (Recall + Precision)

# In[23]:


from sklearn.metrics import f1_score

# We use SV model, because is the model with better precision and performance in this case
f1_score(test_y, svc.predict(test_x_vector),
         labels=['positive', 'negative'],
         average=None)

# Te result tell us how good is our model
# f1_score goes form 0 to 1


# # 4.3 Classification report

# In[24]:


from sklearn.metrics import classification_report

# We pass the true data and those that we are going to predict
print(classification_report(test_y, 
                            svc.predict(test_x_vector),
                            labels=['positive', 'negative']))


# # 4.4 Confusion Matrix

# A confusion matrix is a table with two rows and two columns that reports the number of false positives, false negatives, true positives, and true negatives

# In[25]:


from sklearn.metrics import confusion_matrix

confusion_matrix(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'])


# # 5 Optimizing the Model

# # 5.1 GridSearchCV

# An exhaustive search will be made of which parameters within those that we specify are the best for our model. The model will be optimized.

# In[26]:


from sklearn.model_selection import GridSearchCV

# C -> Penalty parameter, error term, indicates to the algorithm optimization how much error is bearable
# Kernel -> Part of the system that does all the processing, in it we must specify what type of function we want to use (linear, polynomial, rbf, etc)
parameters = {'C': [1,4,8,16,32] ,'kernel':['linear', 'rbf']}

# We create instances
svc = SVC()

# CV = Crossed Validation = How many validations will be made
svc_grid = GridSearchCV(svc,parameters, cv=5,)

svc_grid.fit(train_x_vector, train_y)


# Best parameter is 4, and kernel rbf

# In[28]:


print(svc_grid.best_estimator_)
print(svc_grid.best_params_)


# In[30]:


print(svc_grid.best_score_)


# References: https://www.youtube.com/watch?v=V4ab6qsJZMY
