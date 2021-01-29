# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 18:07:29 2020

@author: Lenovo
"""

# Twitter Sentiment Analyzer

import tweepy
from IPython.display import Image
import streamlit as st
import nltk
from nltk.corpus import stopwords
import re

from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import cv2
import csv
from skimage import io

import urllib
from textblob import TextBlob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

#st.set_option('deprecation.showPyplotGlobalUse', False)
# Twitter Credentials

consumer_key = 'jDvT8Ydto8ABvgx0JsNqcOZsS'
consumer_secret = 'OTZkuey2d0YNxHIWZDp5O6vZ7A1bhod7KZzf1eOXV1Qxi8OwxG'

access_token = '1249542581660323845-JUclgVDUla2o1S0evqsi9C0sRsRWxF'
access_token_secret = 'FKMJkElmc0tFuGYYDxKCjKa6g6l2OFPL31sHfPZRiOn8v'

authenticate = tweepy.OAuthHandler(consumer_key, consumer_secret) 
authenticate.set_access_token(access_token, access_token_secret) 

api = tweepy.API(authenticate, wait_on_rate_limit = True)

st.title('Twitter Sentiment Analyzer')
images = cv2.imread('twitter5.png')
#cv2.imshow('Image',images)
st.image(images, height = 300, width = 300)
st.subheader("This App gives performs the Sentiment Analysis for any twitter account")

screen_name = st.text_area("Enter the exact twitter handle of the Personality (without @)")

alltweets = []  
    
new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    
alltweets.extend(new_tweets)
    
oldest = alltweets[-1].id - 1
    
while len(new_tweets) > 0:
        
    new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1
            

for j in alltweets:
    
    totaltweets = j.user.statuses_count
    following = j.user.friends_count
    Description = j.user.description
    followers = j.user.followers_count
    Prof_image_url = j.user.profile_image_url


image = io.imread(Prof_image_url)
cv2.imshow("Incorrect", image)
cv2.imshow("Correct", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


twts = []
hshtgs = []
retweets = []

for tweet in alltweets:
    
    twts.append(tweet.text)
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

pd.set_option('display.max_colwidth', None)
#pd.set_option('display.html.use_mathjax',False)
       

dict = {'Tweets': twts, 'Retweets':retweets}  
dfs = pd.DataFrame(dict)

dfs["word_count"] = dfs["Tweets"].apply(lambda tweet: len(tweet.split()))

senti_analyzer = SentimentIntensityAnalyzer()

compound_score = []

for sen in dfs['Tweets']:
    
    compound_score.append(senti_analyzer.polarity_scores(sen)['compound'])
    
dfs['Compound Score'] = compound_score

Sentiment = []

for i in compound_score:
    
    if i >= 0.05:
        
        Sentiment.append('Positive')
        
    elif i > -0.05 and i < 0.05:
        
        Sentiment.append('Neutral')
        
    else:
        
        Sentiment.append('Negative')
        
dfs['Sentiment'] = Sentiment

# Sentiment Distribution

positive_tweets = pd.DataFrame(columns=['Tweets', 'Hashtags', 'Retweets', 'word_count', 'Compound Score',
       'Sentiment'])
negative_tweets = pd.DataFrame(columns=['Tweets', 'Hashtags', 'Retweets', 'word_count', 'Compound Score',
       'Sentiment'])
neutral_tweets = pd.DataFrame(columns=['Tweets', 'Hashtags', 'Retweets', 'word_count', 'Compound Score',
       'Sentiment'])

for i in range(len(dfs['Sentiment'])):
    
    if dfs.iloc[i]['Sentiment'] == 'Positive':
        positive_tweets = positive_tweets.append((dfs.iloc[i]))
        
    elif dfs.iloc[i]['Sentiment'] == 'Negative':
        negative_tweets = negative_tweets.append((dfs.iloc[i]))
        
    else:
        
        neutral_tweets = neutral_tweets.append((dfs.iloc[i]))

pos_count = sum(dfs['Sentiment']=='Positive')
neg_count = sum(dfs['Sentiment']=='Negative')
neu_count = sum(dfs['Sentiment']=='Neutral')

import matplotlib.pyplot as plt
# Pie chart
labels = ['Positive tweets', 'Negative tweets', 'Neutral reviews']
sizes = [pos_count, neg_count, neu_count]
#colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
#explsion
explode = (0.05,0.05,0.05)

# Most Positive Tweet

pos_max = dfs.loc[dfs['Compound Score']==max(dfs['Compound Score'])]

# Most Negative Tweet

neg_max = dfs.loc[dfs['Compound Score']==min(dfs['Compound Score'])]



#gp = dfs.groupby('Sentiment')
#positive_tweets = dfs.groupby('Sentiment').get_group('Positive')

# Negative Tweets
#negative_tweets = dfs.groupby('Sentiment').get_group('Negative')

# Neutral Tweets

#neutral_tweets = dfs.groupby('Sentiment').get_group('Neutral')

# Wordcloud Function

     

# WordCloud - Positive Tweets

#positive_wordcloud = WordCloud(background_color= "white",width=900, height=500).generate(wordcloud(positive_tweets)[0])

# WordCloud - Negative Tweets

#negative_wordcloud = WordCloud(width=900, height=500).generate(wordcloud(negative_tweets)[0])

# WordCloud - Neutral Tweets

#neutral_wordcloud = WordCloud(background_color= "white",width=900, height=500).generate(wordcloud(neutral_tweets)[0])

# Wordcloud - All Tweets

#total_wordcloud = WordCloud( width=900, height=500).generate(wordcloud(dfs)[0])

# Most Frequent Words - Total Tweets


category = [ "Twitter Account Details","Twitter Sentiment Analyzer"]
choice = st.sidebar.selectbox("Select Your Category", category)


if choice == "Twitter Account Details":
    
    #st.write("<---------- Select Your Category")
    
    st.subheader("The functions performed by the Twitter Sentiment Analyzer Web App are :")
    
    st.write("1. Displays the Profile Image of the Twitter Account")
    st.write("2. Displays the Most Recent Tweets of the Twitter Account")
    st.write("3. Description of the Twitter Account")
    st.write("4. Number of Followers of the Twitter Account")
    st.write("5. The total number of tweets sent by this Twitter Account")
    st.write("6. The number of Twitter Accounts followed by this Twitter Account")

    Analyzer_choice = st.selectbox("Select the Activities",  ["Display the Profile Image of the Twitter Account","Display the Most Recent Tweets of the Twitter Account","Description of the Twitter Account" ,"Number of Followers of this Twitter Account","The total number of tweets sent by this Twitter Account", "The number of Twitter Accounts followed by this Twitter Account"])

    
    if st.button("Analyze"):

			
        if Analyzer_choice == "Display the Profile Image of the Twitter Account":
            
            cv2.imshow('Image',image)
            st.image(image, width = 100)


        elif Analyzer_choice == "Display the Most Recent Tweets of the Twitter Account":
                
            st.write("The Most Recent Tweets are:") 
            #st.write(df.shape)
            st.table(dfs.head(7))
                
        elif Analyzer_choice == "Description of the Twitter Account":
        
            st.write("The desrciption of this Twitter Account is :")
            st.write(Description)

        elif Analyzer_choice == "The total number of tweets sent by this Twitter Account":
        
            st.write('The total number of tweets by this Twitter Account are:')
            st.write(totaltweets)

                
        elif Analyzer_choice == "Number of Followers of this Twitter Account":
                
            st.write('The number of followers for this Twitter Account has are: ')
            st.write(followers)
        
        else:
        
            st.write('The number of Twitter Accounts followed by this Twitter Account has are: ') 
            st.write(following)
            
else:
    
            
    def plot_Cloud(wordCloud):
        plt.figure( figsize=(20,10), facecolor='w')
        plt.imshow(wordCloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()

    def wordcloud(data):
    
        words_corpus = ''
        words_list = []

    
        for rev in data["Tweets"]:
        
            text = str(rev).lower()
            text = text.replace('rt', ' ') 
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

        
    at = nltk.FreqDist(wordcloud(dfs)[1])
    dt = pd.DataFrame({'Wordcount': list(at.keys()),
                  'Count': list(at.values())})
# selecting top 10 most frequent hashtags     
    dt = dt.nlargest(columns="Count", n = 10)

# Most Frequent Words - Positive Tweets

    ap = nltk.FreqDist(wordcloud(positive_tweets)[1])
    dp = pd.DataFrame({'Wordcount': list(ap.keys()),
                  'Count': list(ap.values())})
# selecting top 10 most frequent hashtags     
    dp = dp.nlargest(columns="Count", n = 10) 

# Most Frequent Words - Negative Tweets

    an = nltk.FreqDist(wordcloud(negative_tweets)[1])
    dn = pd.DataFrame({'Wordcount': list(an.keys()),
                  'Count': list(an.values())})
# selecting top 10 most frequent hashtags     
    dn = dn.nlargest(columns="Count", n = 10) 

# Most Frequent Words - Neutral Tweets

    au = nltk.FreqDist(wordcloud(neutral_tweets)[1])
    du = pd.DataFrame({'Wordcount': list(au.keys()),
                  'Count': list(au.values())})
# selecting top 10 most frequent hashtags     
    du = du.nlargest(columns="Count", n = 10) 


# WordCloud - Negative Tweets

    

# WordCloud - Neutral Tweets

    #neutral_wordcloud = WordCloud(background_color= "white",width=900, height=500).generate(wordcloud(neutral_tweets)[0])

# Wordcloud - All Tweets

    #total_wordcloud = WordCloud( width=900, height=500).generate(wordcloud(dfs)[0])
    
    
    st.subheader("The functions performed by the Web App are :")

    st.write("1. Displays the most recent tweets")
    st.write("2. Displays the Sentiment Distribution of the tweets")
    st.write("3. Displays the most recent positive tweets")
    st.write("4. Generates a wordcloud for all the positive tweets")
    st.write("5. Displays the most positive tweet")
    st.write("6. Displays the most frequent words used in the positive tweets")

    Senti_Analyzer_choices = st.selectbox("Analysis Choice", ["Display the most recent tweets","Display the Sentiment Distribution of the tweets","Display the most recent positive tweets","Generate a wordcloud for all the positive tweets","Generate a wordcloud for all the neutral tweets","Generate a wordcloud for all the tweets","Display the most positive tweet","Display the most frequent words used in all the tweets","Display the most frequent words used in the positive tweets"])

    if st.button("Analyze"):
	
        if Senti_Analyzer_choices == "Display the most recent tweets":
            
            st.write("The Most Recent Tweets are:") 
            st.table(dfs.head())
            
        elif Senti_Analyzer_choices =="Display the Sentiment Distribution of the tweets":
            
            centre_circle = plt.Circle((0,0),0.70,fc='white')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            plt.tight_layout()
            st.write(plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode))
            st.pyplot(use_container_width=True)
                       
        elif Senti_Analyzer_choices =="Display the most recent positive tweets":
            
            st.write("The Most Recent Positive Tweets are:")
            st.write(positive_tweets.head())

       
        elif Senti_Analyzer_choices =="Generate a wordcloud for all the positive tweets":
            
            positive_wordcloud = WordCloud(background_color= "white",width=900, height=500).generate(wordcloud(positive_tweets)[0])
            st.write(plot_Cloud(positive_wordcloud))
            st.pyplot(use_container_width=True)
            
        elif Senti_Analyzer_choices =="Generate a wordcloud for all the neutral tweets":
            pass
            
        elif Senti_Analyzer_choices =="Generate a wordcloud for all the tweets":
            pass
        
        elif Senti_Analyzer_choices =="Display the most positive tweet":
            
            st.write("The most positive tweet is")     
            st.table(pos_max)
        
        
        elif Senti_Analyzer_choices =="Display the most frequent words used in the positive tweets":
            
            st.write('Most Frequent Words in the positive tweets')
            st.write(sns.barplot(data=dp, x= "Wordcount", y = "Count"))
            st.pyplot(use_container_width=True)
       
        else:
            st.write('Most Frequent Words in all of the tweets')
            st.write(sns.barplot(data=dt, x= "Wordcount", y = "Count"))
            st.pyplot(use_container_width=True)
    
    st.write("1. Displays the most recent negative tweets")
    st.write("2. Generates a wordcloud for all the negative tweets")
    st.write("3. Displays the Histogram for the Word Count Distribution of all the tweets")
    st.write("4. Displays the most negative tweet")
    st.write("5. Displays the most frequent words used in the negative tweets") 
    
    Sent_Analyzer_choices = st.selectbox("Analysis Choices", ["Display the most recent negative tweets","Generate a wordcloud for all the negative tweets","Display the Histogram for the Word Count Distribution of all the tweets","Display the most negative tweet","Display the most frequent words used in the negative tweets"])

       
    if st.button("Analyze."):
      
        if Sent_Analyzer_choices =="Display the most recent negative tweets":
            
            st.write("The Most Recent Positive Tweets are:")
            st.table(negative_tweets.head())
        
        elif Sent_Analyzer_choices =="Generate a wordcloud for all the negative tweets":
            
            negative_wordcloud = WordCloud(width=900, height=500).generate(wordcloud(negative_tweets)[0])
            st.write(plot_Cloud(negative_wordcloud))
            st.pyplot(use_container_width=True)

        elif Sent_Analyzer_choices =="Display the Histogram for the Word Count Distribution of all the tweets":
            
            st.write(sns.distplot(dfs['word_count']))
            st.pyplot(use_container_width=True)
     
        
        elif Sent_Analyzer_choices =="Display the most negative tweet":
            
            st.write("The most negative tweet is")
            st.table(neg_max)
        

        elif Sent_Analyzer_choices =="Display the most frequent words used in the negative tweets":
            
            st.write('Most Frequent Words in the negative tweets')
            st.write(sns.barplot(data=dn, x= "Wordcount", y = "Count"))
            st.pyplot(use_container_width=True)
            
            
            
            






    

        



