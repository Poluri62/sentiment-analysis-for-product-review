from textblob import TextBlob
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_pickle('corpus.pkl')
data.review = data.review.astype(str)
print(type(data.review))

pol = lambda x:TextBlob(x).sentiment.polarity
data['polarity'] = data['review'].apply(pol)
data = data[['index','asin','review','rating','polarity']]
print(data[:10])

data['pred'] = pd.cut(data['polarity'], bins=5, labels=[1,2,3,4,5])
data = data.drop(['polarity'], axis=1)

print(data[:10])

#Calculating accuracy
#accuracy = (len(data.loc[data['rating']==data['pred']])/len(data))*100
accuracy = accuracy_score(data['rating'], data['pred'])*100
print('Accuracy: '+ str(accuracy))

#Calculate f1 score
f1 = f1_score(data['rating'],data['pred'], average = 'macro')*100
print('F1 Score: '+ str(f1))

#print(confusion_matrix(data['rating'], data['pred']))


plt.matshow(confusion_matrix(data['rating'], data['pred']), interpolation='nearest')
plt.title('TextBlob Confusion Matrix')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()