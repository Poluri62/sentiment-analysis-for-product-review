import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_pickle('corpus.pkl')
data.review = data.review.astype(str)
print(type(data.review))

lr_object = Pipeline(
    [
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(solver='liblinear', multi_class='auto')),
    ]
)

'''
X, y = (data['review'].values, data['rating'].values)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
print(X_train)
print(len(X_test))
'''

print(len(data))
data_train = data[:105000]
data_test = data[10500:]

learner = lr_object.fit(data_train['review'], data_train['rating'])
data_test['pred'] = learner.predict(data_test['review'])
data_test = data_test[['index','asin','review','rating','pred']]
print(data_test)
#Calculating accuracy
accuracy = accuracy_score(data_test['rating'], data_test['pred'])*100
print(accuracy)

#Calculate f1 score
f1 = f1_score(data_test['rating'],data_test['pred'], average = 'macro')*100
print(f1)


#Confusion matrix
plt.matshow(confusion_matrix(data_test['rating'], data_test['pred']), interpolation='nearest')
plt.title('Logistic Regression Confusion Matrix')
plt.colorbar()
plt.ylabel('Expected Label')
plt.xlabel('Predicted Label')
plt.show()