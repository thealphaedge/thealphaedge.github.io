---
layout: post
title: Getting Started with NLP!
---

## Hello World
## Bonjour Tout Le Monde
## 世界你好  
  
  
Greetings from AlphaEdge team!
Welcome to our blog. Please spend a few minutes with us getting a glimpse of our project, goal and current progress.
Our team is working on the project of “Bitcoin futures trading strategy based on Natural Language Processing (NLP)”. This is a classroom project of our university course **_Text Analytics and Natural Language Processing in Finance_**. Our goal is to derive a quantitative trading strategy of **bitcoin futures contract** based on bitcoin textual data analysis.
Till now, we have finished collecting data and preprocessing data, we would like to share our experience in this blog and hope this could be useful if you have interests in similar projects!

> Flash Card: Bitcoin Futures
>In December 2017, CME and the Chicago Board Options Exchange(CBOE) launched their respective cash-settled bitcoin futures trading products as one of the first major moves to bring a bitcoin-based (or, in this case, a bitcoin price-based) trading product to the mainstream financial world ([Forbes, 2019](https://www.forbes.com/sites/benjaminpirus/2019/08/28/cme-bitcoin-futures-now-average-370-million-in-trading-per-day/#7315d86667ea)).
  
    
    
## Where to Get Data?  
![Data, data, data](https://miro.medium.com/max/1382/1*XbUHd4PsJgmAY3S0oV-ijA.png)

Data, data, data _Image source: Analytics India Magazine_
  
Here comes the first and one of the most crucial steps of most NLP projects. Where can we get proper textual data? It can be annoying for experienced analysts not to mention us as newbies.
In the beginning, we tried to use Eikon Reuters to extract the bitcoin news articles. However, we found that the official API of Reuters has a limitation of 100 outcomes every time. If there are more than 100 outcomes each time, it will fall to an error and show only 10 of the recent outcomes. With the inspiration of our course instructor, we decided to turn to the Twitter database for easy access. Since the main news sources, such as coin desk, all have their twitter pages and post the news on its timeline with a short summary to present the main idea.
Twitter Inc. provides official API called [Tweepy](https://www.tweepy.org/), with Tweepy we can download tweets from a few Bitcoin news provider’s timelines. Following is a code snippet to download tweets from a user’s timeline.
  
```python

# Import tweepy and csv.
import tweepy 

# Set up twitter API login information.
access_token = 'xxxx'
access_token_secret = 'xxxx'
consumer_key = 'xxxx'
consumer_secret = 'xxxx'

# Authorize tweeter.
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Twitter user you want to download.
screen_name = "the_id_of_the_user"

# Initialize a list to hold all the tweets.
alltweets = []	
# Make initial request for most recent tweets (200 is the maximum allowed count each time).
new_tweets = \
    api.user_timeline(
        screen_name = screen_name,
        count = 200,
        exclude_replies = True)
# Save most recent tweets to the list.
alltweets.extend(new_tweets)
# Save the id of the oldest tweet less one.
oldest = alltweets[-1].id - 1
# Keep grabbing tweets until there are no tweets left to grab.
while len(new_tweets) > 0:
    print("getting tweets before %s" % oldest)
    # All subsequent requests use the max_id param to prevent duplicates.
    new_tweets = \
        api.user_timeline(
            screen_name = screen_name,
            count = 200,
            exclude_replies = True,
            max_id = oldest)
    # Save most recent tweets.
    alltweets.extend(new_tweets)
    # Update the id of the oldest tweet less one.
    oldest = alltweets[-1].id - 1
    print("...%s tweets downloaded so far" % len(alltweets))
```
  
  
Adopting this method, we downloaded 18k+ tweets from Dec 17 to Nov 19, with these data we will be able to generate sentiment score or adopt machine learning method to derive our trading strategy in the following step.
Though we applied Twitter as our main database, there are also other methods which may require more efforts but still can be helpful if you are looking for Bitcoin-related textual data:
* From [NewsAPI](https://newsapi.org/) extract news articles with query q = ‘bitcoin’ 
* Scrape Reddit comments with [PRAW](https://towardsdatascience.com/scraping-reddit-data-1c0af3040768) and then select those about Bitcoin 
* Web scraping to get news from specific websites

In conclusion, there are a huge amount of unstructured text data all over the internet. It can be a hard choice for you: which type of text data is the best. Long articles? Academic reports? Or social media fragments? There is no actually the ‘best’ type of textual data, for beginners, it is always good to start with established tools and try different methods. Along the way, you will find the most suitable type of data for your purpose, and never give up even the result comes out not well. It is science that is built from failures.

![cute scientist kid](https://www.incimages.com/uploaded_files/image/970x450/081312_Science_Fail_1725x810-PAN_19604.jpg)
Failure can be fun!  _Image source: Getty_


## Next Up…
 
Thanks for reading! In our next post, we will talk about Trading strategy backtest in Python. 
Stay tuned!
