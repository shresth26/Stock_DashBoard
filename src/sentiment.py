import nltk
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from src.constants import FINWIZ_URL, TICKERS

nltk.download('vader_lexicon')


class NewsSentimentData:

    @staticmethod
    def collect_news_data():

        print("Get news tables")
        news_tables = NewsSentimentData.get_news_tables(TICKERS)

        print("Parse news data")

        parsed_news = NewsSentimentData.parse_news_data(news_tables)

        print("Get news sentiment")

        sentiment_data = NewsSentimentData.get_sentiment(parsed_news)

        print("draw plots")

        NewsSentimentData.draw_plot(sentiment_data)

        print("Draw Wordcloud")

        NewsSentimentData.draw_wordcloud(sentiment_data)

    @staticmethod
    def get_news_tables(TICKERS: list) -> dict:

        news_tables = {}
        for ticker in TICKERS:
            url = FINWIZ_URL + ticker
            req = Request(url=url, headers={'user-agent': 'my-app/0.0.1'})
            response = urlopen(req)
            html = BeautifulSoup(response, features="lxml")
            news_table = html.find(id='news-table')
            news_tables[ticker] = news_table

        return news_tables

    @staticmethod
    def parse_news_data(news_tables: dict) -> list:

        parsed_news = []

        for file_name, news_table in news_tables.items():
            for x in news_table.findAll('tr'):
                text = x.a.get_text()
                date_scrape = x.td.text.split()

                if len(date_scrape) == 1:
                    time = date_scrape[0]
                else:
                    date = date_scrape[0]
                    time = date_scrape[1]

                ticker = file_name.split('_')[0]
                parsed_news.append([ticker, date, time, text])

        return parsed_news

    @staticmethod
    def get_sentiment(parsed_news: list) -> pd.DataFrame:

        vader = SentimentIntensityAnalyzer()

        columns = ['Ticker', 'Date', 'Time', 'Headline']
        parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)
        scores = parsed_and_scored_news['Headline'].apply(vader.polarity_scores).tolist()

        scores_df = pd.DataFrame(scores)
        parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')
        parsed_and_scored_news['Date'] = pd.to_datetime(parsed_and_scored_news.Date).dt.date

        return parsed_and_scored_news

    @staticmethod
    def draw_plot(sentiment_data: pd.DataFrame) -> None:

        plt.rcParams['figure.figsize'] = [10, 6]
        mean_scores = sentiment_data.groupby(['Ticker', 'Date']).mean()

        mean_scores = mean_scores.unstack()

        mean_scores = mean_scores.xs('compound', axis="columns").transpose()

        mean_scores.plot(kind='bar')
        plt.show()

    @staticmethod
    def draw_wordcloud(sentiment_data: pd.DataFrame):

        text = " ".join(i for i in sentiment_data.Headline)
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
