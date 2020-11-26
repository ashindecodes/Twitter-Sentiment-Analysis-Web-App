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


https://medium.com/python-in-plain-english/scraping-tweets-with-tweepy-python-59413046e788


# In[ ]:


https://github.com/harit198/Tweet-Analyzer/blob/master/app.py


# In[ ]:


https://www.youtube.com/watch?v=MqIw68fEq1k


# In[ ]:


https://www.youtube.com/watch?v=ujId4ipkBio&feature=emb_logo


# In[5]:


# Twitter Credentials

consumer_key = 'jDvT8Ydto8ABvgx0JsNqcOZsS'
consumer_secret = 'OTZkuey2d0YNxHIWZDp5O6vZ7A1bhod7KZzf1eOXV1Qxi8OwxG'

access_token = '1249542581660323845-JUclgVDUla2o1S0evqsi9C0sRsRWxF'
access_token_secret = 'FKMJkElmc0tFuGYYDxKCjKa6g6l2OFPL31sHfPZRiOn8v'


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


# In[22]:


df1


# In[ ]:



# Lowercase
# Removing '@....' , 'https://.....' ,  '#.....' i.e. Remove ‘RT’, UserMentions and links
      "RT.....:"
# Remove numbers
# Remove Hashtags, urls, mentions and @ from the tweets
    
# Removing spaces  
    https://www.youtube.com/watch?v=Otde6VGvhWM&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=11

#Remove punctuation marks and special characters

#Removing stopwords


# In[23]:


df1['Tweets']


# In[ ]:


# Removing urls
tw = [re.sub(r'http\S+', '', x)for x in df1['Tweets']]

for i in df1['Tweets']:
    
    


# In[ ]:





# In[24]:


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob
import re


# In[ ]:


# Remove Stopwords & Lemmitization


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


# In[ ]:


# Stemming of words 
    
    Process of reducing the infected words to their stem word
    Problem with stemming is that the representations converted to their stem words don't have any meaning
    and this disadvantage is taken care of in Lemmitization
    
    
# Lemmitization

    - Lemmitization also does the same thing as stemming but it converts the words
    to a stem word which is meaningful. Lemmitization takes more time
    
      
# Bag of Words

    1.Binary Bag of Word
    2.Bag of Word
    
    - In order to solve the disdvantages of Bag of Word technique we have TF-IDF
     
# TF-IDF

            
      - TF - No of repitions of words in sentences/No of words in sentences

        TF
        sent1   sent2   sent3
        
good     1/2      1/2     1/3
boy      1/2      0       1/3
girl     0        1/2     1/3

        IDF
    
words   IDF

good    log(3/3)=log(1)=0
boy     log(3/2)
girl    log(3/2)


                IDF - log(No of sentences/No of sentences containing that word)

                We calculate TFxIDF for every word in all the sentences

            good          boy          girl
    
Sent1        0      1/2xlog(3/2)         0

Sent2        0             0           1/2xlog(3/2)

Sent3        0      1/3xlog(3/2)      1/3xlog(3/2)

# Word2Vec    
        
         - Try GOOGLE Word2Vec
        
        
        - Word2Vec is applied on huge amount of data
        - In both Bag of Words and TF IDF semantic information is not stored. 
        - TF-IDF gives importance to uncommon words
        - There is definitely a chance of over-fitting
    
    
        - In a Word2Vec model, each word is basically represented as a vector of 32
        or more dimension instead of a single number.
        
        - Also in a Word2Vec model, the semantic information and relation between 
        different words is also preserved.
        
        
        Steps to create Word2Vec
        
        
        - Tokenisation of sentences
        
        - Create Histograms
        
        - Take Most Frequent Words
        
        - Create a matrix with all the unique words.
          It also represents the occurence relation between the words.


# In[ ]:





# # Other NLP Projects
# 
# https://machinelearningknowledge.ai/natural-language-processing-github-projects-to-inspire-you/
#     
# + Automatic Summarization of Scientific Papers
# 
# + Toxic Comments Classification (Social Media)

# # Word Embeddings
# 
# + https://www.youtube.com/watch?v=5PL0TmQhItY
#     
# + https://www.youtube.com/watch?v=pO_6Jk0QtKw

# # Make changes to the Amazon Product Reviews Sentiment Analyzer like tokenization, lemmitization, NLP tasks.....

# # Deploying ML Models on Mobile Apps? 
# 
# https://medium.com/@BdourEbrahim/how-to-integrate-machine-learning-models-into-your-app-b77717e2702

# # Deploying WebApp
# 
# https://www.kdnuggets.com/2020/10/deploying-streamlit-apps-streamlit-sharing.html
# 
# https://towardsdatascience.com/quickly-build-and-deploy-an-application-with-streamlit-988ca08c7e83

# # Fake News Classifier Web App NLP ?
# 
# https://www.youtube.com/watch?v=MO5n5JaRotc
# 
# https://www.youtube.com/watch?v=MXPh_lMRwAI

# # NLP Playlist Krish Naik
# 
# + https://www.youtube.com/watch?v=fM4qTMfCoak&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm
# 
# + https://www.youtube.com/watch?v=kYXiC6jPiB4
# 
# + https://www.youtube.com/watch?v=AnvrJNLKp0k

# # Vader Sentiment
# 
# + https://www.youtube.com/watch?v=AnvrJNLKp0k

# # Stemming and Lemmitization, nltk
# 
# + https://www.youtube.com/watch?v=kYXiC6jPiB4
# 

# # Stemming and Lemmitization, NLP
# 
# + https://www.youtube.com/watch?v=fM4qTMfCoak&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm

# In[ ]:




