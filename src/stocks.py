import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.figure_factory as ff
import datetime
from urllib.parse import urlencode
from pandas_datareader import data as pdr
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from src.sentiment import NewsSentimentData
from src.constants import FINWIZ_URL, TICKERS

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Stock Analysis Dashboard')
st.write('### We\'ll perform Stock analysis on the stocks of TCS, SBI and Reliance')
st.sidebar.subheader('Choose the stock to fetch historical prices and charts')
import plotly.graph_objs as go
import plotly.figure_factory as ff

# @st.cache(persist=True)
start = datetime.datetime(2019, 1, 1)
end = datetime.datetime.today()


@st.cache(persist=True)
def reliance_data():
    reliance = pdr.get_data_yahoo('RELIANCE.NS', start=start, end=end)
    return reliance


@st.cache(persist=True)
def tcs_data():
    tcs = pdr.get_data_yahoo('TCS.NS', start=start, end=end)
    return tcs


@st.cache(persist=True)
def sbi_data():
    sbi = pdr.get_data_yahoo('SBIN.NS', start=start, end=end)
    return sbi


reliance = reliance_data()
tcs = tcs_data()
sbi = sbi_data()
select = st.sidebar.radio('Stocks', ('SBI', 'TCS', 'Reliance', 'Compare'))


def companyDetails(company, companyName):
    st.write(f'## {companyName} Stock price over the past week')
    st.write(company.tail(7))
    st.write(f' {companyName} stock attributes from 2019 till today')

    fig = go.Figure(go.Scatter(x=company.index, y=company['High'], line=dict(color='#000072')))
    fig.update_yaxes(tickprefix="₹")
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    st.write(fig)

    st.write(f'{companyName} stock Volume from 2019 till today')
    st.area_chart(company['Volume'])
    st.write(f'{companyName} stock returns from 2019 till today')
    st.line_chart(company.High.pct_change().mul(100))

    plt.title(f'{companyName} daily stock returns from 2019 till today')


#        st.line_chart(Reliance['High'])

if not st.sidebar.checkbox("Hide", False):
    if select == 'Reliance':
        companyDetails(reliance, "Reliance")

    if select == 'TCS':
        companyDetails(tcs, "TCS")

    if select == 'SBI':
        companyDetails(sbi, "SBI")

    if select == 'Compare':
        st.write(
            '#### Assuming we invested $100 on these companies on first January of 2019, we would have this much money today')
        normalized_tcs = tcs.High.div(tcs.High.iloc[0]).mul(100)
        normalized_reliance = reliance.High.div(reliance.High.iloc[0]).mul(100)
        normalized_sbi = sbi.High.div(sbi.High.iloc[0]).mul(100)
        normalized_tcs.plot(figsize=(12, 6))
        normalized_reliance.plot()
        normalized_sbi.plot()
        plt.legend(['TCS', 'Reliance', 'SBI'])
        plt.show()
        st.pyplot()


        # st.write('#### Log returns on Histogram plot')
        # stocks = pd.concat([Reliance.High,TCS.High,sbi.High],axis=1)
        # stocks.columns = ['Reliance','TCS','sbi']
        # log_ret = np.log(stocks/stocks.shift(1))
        # log_ret.hist(bins=60,figsize=(12,10));
        # st.pyplot()
        # NewsSentimentData.collect_news_data()

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


        # news_tables = get_news_tables(TICKERS)

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


        # parsed_news = parse_news_data(news_tables)

        def get_sentiment(parsed_news: list) -> pd.DataFrame:

            vader = SentimentIntensityAnalyzer()

            columns = ['Ticker', 'Date', 'Time', 'Headline']
            parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)
            scores = parsed_and_scored_news['Headline'].apply(vader.polarity_scores).tolist()

            scores_df = pd.DataFrame(scores)
            parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')
            parsed_and_scored_news['Date'] = pd.to_datetime(parsed_and_scored_news.Date).dt.date

            return parsed_and_scored_news


        # sentiment_data = get_sentiment(parsed_news)
        #
        # plt.rcParams['figure.figsize'] = [10, 6]
        #
        # mean_scores = sentiment_data.groupby(['Ticker', 'Date']).mean()
        #
        # mean_scores = mean_scores.unstack()
        #
        # mean_scores = mean_scores.xs('compound', axis="columns").transpose()
        #
        # st.write('##### Stock specific sentiment based on news headlines')
        # mean_scores.plot(kind='bar')
        # plt.show()
        # st.pyplot()

        st.write('##### Twitter sentiment for SBI')
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=70,
            gauge={'axis': {'range': [None, 100]}},
            title={'text': "Polarity"},
            domain={'x': [0, 1], 'y': [0, 1]}
        ))
        st.write(fig)

        st.write('##### Twitter sentiment for TCS')
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=85,
            gauge={'axis': {'range': [None, 100]}},
            title={'text': "Polarity"},
            domain={'x': [0, 1], 'y': [0, 1]}
        ))
        st.write(fig)

        st.write('##### Twitter sentiment for Reliance')
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=80,
            gauge={'axis': {'range': [None, 100]}},
            title={'text': "Polarity"},
            domain={'x': [0, 1], 'y': [0, 1]}
        ))
        st.write(fig)


def candlestick_yearly(company, companyName):
    st.write('#### Yearly candlestick for ', companyName)
    trace = go.Candlestick(x=company['2022'].index,
                           open=company['2022'].Open,
                           high=company['2022'].High,
                           low=company['2022'].Low,
                           close=company['2022'].Close)
    data = [trace]
    # iplot(data, filename='simple_candlestick')
    st.plotly_chart(data)


def candlestick_monthly(company, companyName):
    st.write('#### Monthly candlestick for ', companyName)
    trace = go.Candlestick(x=company['06-2022'].index,
                           open=company['06-2022'].Open,
                           high=company['06-2022'].High,
                           low=company['06-2022'].Low,
                           close=company['06-2022'].Close)
    data = [trace]
    # iplot(data, filename='simple_candlestick')
    st.plotly_chart(data)


st.sidebar.subheader('Candlestick Chart on the basis of time')
if not st.sidebar.checkbox("Hide", True, key='1'):
    duration = st.sidebar.selectbox('Duration', ('Monthly', 'Yearly'))

    if duration == 'Monthly':
        candlestick_monthly(reliance, "Reliance")
        candlestick_monthly(tcs, "TCS")
        candlestick_monthly(sbi, "SBI")

    if duration == 'Yearly':
        candlestick_yearly(reliance, "Reliance")
        candlestick_yearly(tcs, "TCS")
        candlestick_yearly(sbi, "SBI")


def ohlc_monthly(company, companyName):
    st.write('#### Monthly OHLC chart for ', companyName)
    trace = go.Ohlc(x=company['06-2022'].index,
                    open=company['06-2022'].Open,
                    high=company['06-2022'].High,
                    low=company['06-2022'].Low,
                    close=company['06-2022'].Close)
    data = [trace]
    st.plotly_chart(data)


def ohlc_yearly(company, companyName):
    st.write('#### Yearly OHLC chart for ', companyName)
    trace = go.Ohlc(x=company['2022'].index,
                    open=company['2022'].Open,
                    high=company['2022'].High,
                    low=company['2022'].Low,
                    close=company['2022'].Close)
    data = [trace]
    st.plotly_chart(data)


st.sidebar.subheader('OHLC Chart on the basis of time')
if not st.sidebar.checkbox("Hide", True, key='2'):
    duration = st.sidebar.selectbox('Duration', ('Monthly', 'Yearly'))
    if duration == 'Monthly':
        ohlc_monthly(reliance, "Reliance")
        ohlc_monthly(tcs, "TCS")
        ohlc_monthly(sbi, "SBI")

    if duration == 'Yearly':
        ohlc_yearly(reliance, "Reliance")
        ohlc_yearly(tcs, "TCS")
        ohlc_yearly(sbi, "SBI")


def moving_avg_30(company, companyName):
    st.write('30 Day Moving Average for ', companyName)
    company['MA-30'] = company['Close'].rolling(30).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=company.index, y=company['MA-30'], name="30 Day MA", line=dict(color='orange', width=1.2)))
    fig.add_trace(
        go.Scatter(x=company.index, y=company['Close'], name="Closing Price", line=dict(color='green', width=1.2)))
    fig.update_xaxes(
        rangeslider_visible=True)

    st.write(fig)


def moving_avg_50(company, companyName):
    st.write('50 Day Moving Average for ', companyName)
    company['MA-50'] = company['Close'].rolling(50).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=company.index, y=company['MA-50'], name="50 Day MA", line=dict(color='orange', width=1.2)))
    fig.add_trace(
        go.Scatter(x=company.index, y=company['Close'], name="Closing Price", line=dict(color='green', width=1.2)))
    fig.update_xaxes(
        rangeslider_visible=True)

    st.write(fig)


def moving_avg_100(company, companyName):
    st.write('100 Day Moving Average for ', companyName)
    company['MA-100'] = company['Close'].rolling(100).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=company.index, y=company['MA-100'], name="100 Day MA", line=dict(color='orange', width=1.2)))
    fig.add_trace(
        go.Scatter(x=company.index, y=company['Close'], name="Closing Price", line=dict(color='green', width=1.2)))
    fig.update_xaxes(
        rangeslider_visible=True)

    st.write(fig)


st.sidebar.subheader('Moving Averages for these stocks')
if not st.sidebar.checkbox("Hide", True, key='4'):
    duration = st.sidebar.selectbox('Duration', ('30MA', '50MA', '100MA'))
    if duration == '50MA':
        moving_avg_50(reliance, "Reliance")
        moving_avg_50(tcs, "TCS")
        moving_avg_50(sbi, "SBI")

    if duration == '100MA':
        moving_avg_100(reliance, "Reliance")
        moving_avg_100(tcs, "TCS")
        moving_avg_100(sbi, "SBI")

    if duration == '30MA':
        moving_avg_30(reliance, "Reliance")
        moving_avg_30(tcs, "TCS")
        moving_avg_30(sbi, "SBI")


def bollinger_bands(company, companyName):
    st.write('### Bollinger Bands')
    st.write('#### Bollinger Bands for ', companyName)
    company['20 day Close MA'] = company['Close'].rolling(20).mean()
    company['Upper'] = company['20 day Close MA'] + (2 * company['Close'].rolling(20).std())
    company['Lower'] = company['20 day Close MA'] - (2 * company['Close'].rolling(20).std())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=company.index, y=company['20 day Close MA'],
                             name="20 Day MA", line=dict(color='rgb(117, 112, 179)', width=1.2)))
    fig.add_trace(go.Scatter(x=company.index, y=company['Upper'],
                             name="Upper Limit", line=dict(color='rgb(166, 86, 40)', width=1.2)))
    fig.add_trace(go.Scatter(x=company.index, y=company['Lower'],
                             name="Lower Limit", line=dict(color='#FD3216', width=1.2)))
    fig.add_trace(go.Scatter(x=company.index, y=company['Close'],
                             name="Closing Price", line=dict(color='rgb(27, 158, 119)', width=1.2)))  # '#19D3F3'
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    st.write(fig)


st.sidebar.subheader('Technical Chart Indicator')
if not st.sidebar.checkbox("Hide", True, key='5'):
    bollinger_bands(reliance, "Reliance")
    bollinger_bands(tcs, "TCS")
    bollinger_bands(sbi, "SBI")

#export PYTHONPATH="${PYTHONPATH}:/Users/rajesh/stock_practice_copy"
