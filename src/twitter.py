import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
from src.clients.twitter_client import TwitterClient


class TwitterSentimentData:

    @staticmethod
    def collect_twitter_data():

		print("Fetching the tweets list")

        tweets_list = TwitterSentimentData.get_tweets("Tesla")

		print("Getting sentiment for the tweets")

		positive_score, negative_score = TwitterSentimentData.get_tweeet_sentiments(tweets_list)


    @staticmethod
    def get_tweets(ticker: str) -> list:

        tweets: list = []
        search_query = ticker

        try:

            for tweet_object in tweepy.Cursor(TwitterClient.get_twitter_client(), q=search_query + " -filter:retweets",
                                              lang='en', result_type='recent').items(100):
                tweets.append(tweet_object.text)

        except Exception as exc:
            print(f"Unable to fetch tweets due to - {exc}")

		return tweets

	def get_tweet_sentiments(tweets_list: list):

		positive = 0
		negative = 0
		positive_aggregate = 0
		negative_aggregate = 0

		for tweet in tweets_list:
			analysis = TextBlob(tweet)

			if analysis.sentiment.polarity >= 0.2:
				positive+=1

			elif analysis.sentiment.polarity <= -0.2:
				negative +=1

		if positive == 0 and negative == 0:
			positive_aggregate = 0.50
			negative_aggregate = 0.50

		else:
			positive_aggregate = round(positive / (positive + negative),2)
			negative_aggregate = round(negative / (positive + negative),2)

		return positive_aggregate, negative_aggregate


if __name__ == "__main__":
    # calling main function
    main()
