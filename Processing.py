import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import gzip
import json
import pickle

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

#169781 digital musict item
df = getDF('C:\\Users\\Divya\\Desktop\\Project Data\\Digital_Music_5.json.gz')
df = df.drop(['verified','reviewTime', 'reviewerID', 'summary','unixReviewTime','vote','image','style','reviewerName'], axis=1)
#print(df_vdgames)
df = df.rename(columns={"overall": "rating", "reviewText": "review"})
df = df.sample(frac=1).reset_index(drop=True)
#print(len(df))

data = pd.read_pickle('corpus.pkl')
data.review = data.review.astype(str)
df.review = df.review.astype(str)
print(type(data.review))

lr_object = Pipeline(
    [
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(solver='liblinear', multi_class='auto')),
    ]
)

print(len(data))
data_train = data[:105000]
#data_test = data[10500:]

learner = lr_object.fit(data_train['review'], data_train['rating'])
df['pred'] = learner.predict(df['review'])
df = df[['asin','review','rating','pred']]
#print(data_test)

num_of_ratings = []
for i in range(5):
  num_of_ratings.append(len(df.loc[df['pred'] == i+1]))
print(num_of_ratings)


plt.rcParams['figure.figsize'] = [8,6]

x = ['1','2','3','4','5']
y = num_of_ratings
plt.bar(x,y,color='blue')
plt.title('Ratings distribution', fontsize = 20)
plt.xlabel('Rating', fontsize = 15)
plt.ylabel('num_of_ratings', fontsize = 15)
plt.show()