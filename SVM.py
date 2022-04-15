import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_pickle('corpus.pkl')
data.review = data.review.astype(str)
print(type(data.review))

svm_object = Pipeline(
    [
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(
            loss='hinge',
            penalty='l2',
            alpha=1e-3,
            random_state=42,
            max_iter=100,
            learning_rate='optimal',
            tol=None,
        )),
    ]
)

data_train = data[:105000]
data_test = data[105000:]

learner = svm_object.fit(data_train['review'], data_train['rating'])

# Predict class labels using the learner and output DataFrame
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
plt.title('SVM Confusion Matrix')
plt.colorbar()
plt.ylabel('Expected label')
plt.xlabel('Predicted Label')
plt.show()