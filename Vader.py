from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_pickle('corpus.pkl')
data.review = data.review.astype(str)
print(type(data.review))

vader_obj = SentimentIntensityAnalyzer()
compound = lambda x:vader_obj.polarity_scores(x)['compound']
#print(data['review'].apply(pol))
data['compound'] = data['review'].apply(compound)
data = data[['index','asin','review','rating','compound']]
#print(data[:10])

data['pred'] = pd.cut(data['compound'], bins=5, labels=[1,2,3,4,5])
data = data.drop(['compound'], axis=1)

print(data[:10])

#Calculating accuracy
accuracy = accuracy_score(data['rating'], data['pred'])*100
print(accuracy)

#Calculate f1 score
f1 = f1_score(data['rating'],data['pred'], average = 'macro')*100
print(f1)

#Confusion matrix
plt.matshow(confusion_matrix(data['rating'], data['pred']), interpolation='nearest')
plt.title('VADER Confusion Matrix')
plt.colorbar()
plt.ylabel('Expected Label')
plt.xlabel('Predicted Label')
plt.show()