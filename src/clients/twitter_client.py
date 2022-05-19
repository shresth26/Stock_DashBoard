import tweepy
from tweepy import OAuthHandler


class TwitterClient:

    __api = None

    def __init__(self):

        if TwitterClient.__api is not None:
            print(" Already Created Twitter Client")

        else:
            self.create_client()

    @staticmethod
    def create_client():

        consumer_key = 'XXXXXXXXXXXXXXXXXXXXXXXX'
        consumer_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXX'
        access_token = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXX'
        access_token_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXX'

        # attempt authentication
        try:
            # create OAuthHandler object
            auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            TwitterClient.__api = tweepy.API(auth)

        except Exception as exc:
            print(f"Error: Authentication Failed due to  - {exc}")

    @staticmethod
    def get_twitter_client():

        if TwitterClient.__api is None:
            TwitterClient()
        return TwitterClient.__api
