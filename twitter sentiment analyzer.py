#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tweepy


# In[4]:


from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns


# In[ ]:

# Twitter Credentials

consumer_key = ##############
consumer_secret = #############

access_token = ##############
access_token_secret = ##############


# In[6]:


authenticate = tweepy.OAuthHandler(consumer_key, consumer_secret) 


# In[7]:


authenticate.set_access_token(access_token, access_token_secret) 


# In[8]:


api = tweepy.API(authenticate, wait_on_rate_limit = True)


# In[9]:


twitter = api.user_timeline(screen_name='nytimes', count = 100, lang ="en", tweet_mode="extended")


# In[10]:


for j in twitter:
    
    #print("retweet count is :", j.retweet_count)
    #print("tweet is :", j.full_text)
    totaltweets = j.user.statuses_count
    following = j.user.friends_count
    Description = j.user.description
    followers = j.user.followers_count
    Prof_image_url = j.user.profile_image_url
    


# ## Twitter Account Details

# In[11]:


print("Desrciption is :",Description)


# In[12]:


print("Profile image url is :", Prof_image_url)
from IPython.display import Image
Profile_Image = Image(Prof_image_url)
Profile_Image


# In[13]:


print('The number of followers this Twitter Account has are: ', followers)


# In[14]:


print('The number of Twitter Accounts followed by this Twitter Account has are: ', following)


# In[15]:


print('The total number of tweets by this Twitter Account are:', totaltweets)


# In[16]:


twts = []
hshtgs = []
retweets = []

for tweet in twitter:

    twts.append(tweet.full_text)
    retweets.append(tweet.retweet_count)
    hst = tweet.entities['hashtags']
    if len(hst) != 0:
        
        hh_list = []
        for i in range(len(hst)):
            dct = hst[i]
            hh_list.append(str(dct.get('text')))
        hshtgs.append(hh_list)
        
    else:
        hshtgs.append([])
        


# In[17]:


dict = {'Tweets': twts, 'Hashtags': hshtgs, 'Retweets':retweets}  


# In[18]:


df = pd.DataFrame(dict)


# In[19]:


pd.set_option('display.max_colwidth', None)


# In[20]:


df.head(15)


# In[21]:


df1 = df.head(15)


# In[ ]:


Twitter Sentiment Analyzer Functions

 Show recent tweets
# Sentiment Distribution - Pos/Neg/Neu Tweets
# Positive Tweets

# Most Positive Tweet
# Most Negative Tweet
# Negative tweets

# Word Frequency for Positive tweeets - Top 20 words
# Word Frequency for Negative tweets - Top 20 words

# Word Frequent words in Positive and Negative Tweets

# Wordcloud
# Most Frequent Words - Barchart
# Most Frequent Hashtags - Barchart and Wordcloud
# Histogram for Word Count - sns.distplot()


# In[24]:


import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import re


# In[ ]:

# # ---------------------------------------------------------------------------------------------------------------

# In[25]:


# Word Count Distribution Histogram

df1["word_count"] = df1["Tweets"].apply(lambda tweet: len(tweet.split()))
sns.distplot(df1['word_count'])


# In[27]:


df2 = df1


# In[28]:


# Sentiment

Polarity_score = [round(TextBlob(twt).sentiment.polarity, 3) for twt in df2['Tweets']]

df2['Polarity']= Polarity_score

sentiment = ['positive' if score > 0 
                             else 'negative' if score < 0 
                                 else 'neutral' 
                                     for score in Polarity_score]
df2['Sentiment'] = sentiment


# In[29]:


df2.head()


# In[30]:


# Sentiment Distribution

pos_count = sum(df2['Sentiment']=='positive')
neg_count = sum(df2['Sentiment']=='negative')
neu_count = sum(df2['Sentiment']=='neutral')

import matplotlib.pyplot as plt
# Pie chart
labels = ['Positive tweets', 'Negative tweets', 'Neutral reviews']
sizes = [pos_count, neg_count, neu_count]
#colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
#explsion
explode = (0.05,0.05,0.05)
 
    
plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.tight_layout()
plt.show()


# In[31]:


# Most Positive Tweet

pos_max = df2.loc[df2['Polarity']==max(df2['Polarity'])]
pos_max


# In[32]:


# Most Negative Tweet

neg_max = df2.loc[df2['Polarity']==min(df2['Polarity'])]
neg_max


# In[33]:


# Positive Tweets

gp = df2.groupby(by=['Sentiment'])
positive_tweets = gp.get_group('positive')
positive_tweets.head()


# In[34]:


# Negative Tweets

negative_tweets = gp.get_group('negative')
negative_tweets.head()


# In[35]:


# Wordcloud Function

def plot_Cloud(wordCloud):
    plt.figure( figsize=(20,10), facecolor='w')
    plt.imshow(wordCloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


# In[36]:


def wordcloud(data):
    
    words_corpus = ''
    words_list = []

    
    for rev in data["Tweets"]:
        
        text = str(rev).lower()
        text = re.sub(r"http\S+", "", text)        
        text = re.sub(r'[^\w\s]','',text)
        text = ''.join([i for i in text if not i.isdigit()])
        
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        
        # Remove aplha numeric characters
        
        for words in tokens:
            
            words_corpus = words_corpus + words + " "
            words_list.append(words)
            
    return words_corpus, words_list


# In[37]:


# WordCloud - Positive Tweets

from wordcloud import WordCloud
positive_wordcloud = WordCloud(width=900, height=500).generate(wordcloud(positive_tweets)[0])
    
plot_Cloud(positive_wordcloud)


# In[38]:


# WordCloud - Negative Tweets

negative_wordcloud = WordCloud(width=900, height=500).generate(wordcloud(negative_tweets)[0])

plot_Cloud(negative_wordcloud)


# In[39]:


# Wordcloud - Total Tweets

total_wordcloud = WordCloud(width=900, height=500).generate(wordcloud(df2)[0])

plot_Cloud(total_wordcloud)


# In[40]:


# Most Frequent Words - Total Tweets

aa = nltk.FreqDist(wordcloud(df2)[1])
dd = pd.DataFrame({'Wordcount': list(aa.keys()),
                  'Count': list(aa.values())})
# selecting top 10 most frequent hashtags     
dd = dd.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(19,19))
plt.title('Most Frequent Words in all of the tweets')
ax = sns.barplot(data=dd, x= "Wordcount", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# In[41]:


# Most Frequent Words - Positive Tweets

aa = nltk.FreqDist(wordcloud(positive_tweets)[1])
dd = pd.DataFrame({'Wordcount': list(aa.keys()),
                  'Count': list(aa.values())})
# selecting top 10 most frequent hashtags     
dd = dd.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(19,19))
plt.title('Most Frequent Words in the positive tweets')
ax = sns.barplot(data=dd, x= "Wordcount", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# In[52]:


# Most Frequent Words - Negative Tweets

aa = nltk.FreqDist(wordcloud(negative_tweets)[1])
dd = pd.DataFrame({'Wordcount': list(aa.keys()),
                  'Count': list(aa.values())})
# selecting top 10 most frequent hashtags     
dd = dd.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(19,19))
plt.title('Most Frequent Words in the negative tweets')
ax = sns.barplot(data=dd, x= "Wordcount", y = "Count")
ax.set(ylabel = 'Count')
plt.show()




