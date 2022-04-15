import pandas as pd
import gzip
import json
import pickle
import matplotlib.pyplot as plt

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

#3176 fashion items
df_fashion = getDF('C:\\Users\\Divya\\Desktop\\Project Data\\AMAZON_FASHION_5.json.gz')

#2277 appliances
df_appliances = getDF('C:\\Users\\Divya\\Desktop\\Project Data\\Appliances_5.json.gz')

#2375
df_magazine = getDF('C:\\Users\\Divya\\Desktop\\Project Data\\Magazine_Subscriptions_5.json.gz')

#10261
df_music = getDF('C:\\Users\\Divya\\Desktop\\Project Data\\reviews_Musical_Instruments_5.json.gz')

#12805
df_software = getDF('C:\\Users\\Divya\\Desktop\\Project Data\\Software_5.json.gz')

#5269 beauty items
df_beauty = getDF('C:\\Users\\Divya\\Desktop\\Project Data\\All_Beauty_5.json.gz')

#169781 digital musict item
df_digimusic = getDF('C:\\Users\\Divya\\Desktop\\Project Data\\Digital_Music_5.json.gz')

#77071 indsc item
df_indsc = getDF('C:\\Users\\Divya\\Desktop\\Project Data\\Industrial_and_Scientific_5.json.gz')

#34278 luxbeauty item
df_luxbeauty = getDF('C:\\Users\\Divya\\Desktop\\Project Data\\Luxury_Beauty_5.json.gz')

#497577 vdgames item
df_vdgames = getDF('C:\\Users\\Divya\\Desktop\\Project Data\\Video_Games_5.json.gz')

#print(df_fashion[:10])
#print(df_appliances[:10])
#print(df_magazine[:10])
#print(df_music[:10])
#print(df_software[:10])
#print(df_beauty[:10])
#print(df_digimusic[:10])
#print(df_indsc[:10])
#print(df_luxbeauty[:10])
#print(df_vdgames[:10])

df_fashion = df_fashion.drop(['verified','reviewTime', 'reviewerID', 'summary','unixReviewTime','vote','image','style','reviewerName'], axis=1)
#print(df_fashion)

df_appliances = df_appliances.drop(['verified','reviewTime', 'reviewerID', 'summary','unixReviewTime','vote','image','style','reviewerName'], axis=1)
#print(df_appliances)

df_magazine = df_magazine.drop(['verified','reviewTime', 'reviewerID', 'summary','unixReviewTime','vote','image','style','reviewerName'], axis=1)
#print(df_magazine)

df_music = df_music.drop(['reviewTime', 'reviewerID','unixReviewTime','reviewerName','summary','helpful'], axis=1)
df_music = df_music[['overall','asin','reviewText']]
#print(df_music)

df_software = df_software.drop(['verified','reviewTime', 'reviewerID', 'summary','unixReviewTime','vote','image','style','reviewerName'], axis=1)
#print(df_software)

df_beauty = df_beauty.drop(['verified','reviewTime', 'reviewerID', 'summary','unixReviewTime','vote','image','style','reviewerName'], axis=1)
#print(df_beauty)

df_digimusic = df_digimusic.drop(['verified','reviewTime', 'reviewerID', 'summary','unixReviewTime','vote','image','style','reviewerName'], axis=1)
#print(df_digimusic)

df_indsc = df_indsc.drop(['verified','reviewTime', 'reviewerID', 'summary','unixReviewTime','vote','image','style','reviewerName'], axis=1)
#print(df_indsc)

df_luxbeauty = df_luxbeauty.drop(['verified','reviewTime', 'reviewerID', 'summary','unixReviewTime','vote','image','style','reviewerName'], axis=1)
#print(df_luxbeauty)

df_vdgames = df_vdgames.drop(['verified','reviewTime', 'reviewerID', 'summary','unixReviewTime','vote','image','style','reviewerName'], axis=1)
#print(df_vdgames)

df = pd.concat([df_fashion,df_appliances],axis = 0)
df = pd.concat([df,df_magazine],axis = 0)
df = pd.concat([df,df_music],axis = 0)
df = pd.concat([df,df_software],axis = 0)
df = pd.concat([df,df_beauty],axis = 0)
df = pd.concat([df,df_digimusic],axis = 0)
df = pd.concat([df,df_indsc],axis = 0)
df = pd.concat([df,df_luxbeauty],axis = 0)
df = pd.concat([df,df_vdgames],axis = 0)
df = df.reset_index()
df = df.drop(['index'],axis=1)
#print(len(df.loc[df['overall'] == 1.0])) 

df_1 = df.loc[df['overall']==1.0]
df_2 = df.loc[df['overall']==2.0]
df_3 = df.loc[df['overall']==3.0]
df_4 = df.loc[df['overall']==4.0]
df_5 = df.loc[df['overall']==5.0]

df_total = pd.concat([df_1[:30000],df_2[:30000]], axis=0)
df_total = pd.concat([df_total,df_3[:30000]], axis=0)
df_total = pd.concat([df_total,df_4[:30000]], axis=0)
df_total = pd.concat([df_total,df_5[:30000]], axis=0)
df = df_total.reset_index()

df = df.rename(columns={"overall": "rating", "reviewText": "review"})
df = df.sample(frac=1).reset_index(drop=True)
#print(len(df))

df.to_pickle('corpus.pkl')

'''
num_of_ratings = []
for i in range(5):
  num_of_ratings.append(len(df.loc[df['rating'] == i+1]))
print(num_of_ratings)


plt.rcParams['figure.figsize'] = [8,6]

x = ['1','2','3','4','5']
y = num_of_ratings
plt.bar(x,y,color='blue')
plt.title('Ratings distribution', fontsize = 20)
plt.xlabel('Rating', fontsize = 15)
plt.ylabel('num_of_ratings', fontsize = 15)
plt.show()
'''