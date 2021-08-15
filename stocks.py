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
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Stock Analysis Dashboard')
st.write('### We \'ll perform Stock analysis on the stocks of Google, Apple and Microsoft')
st.sidebar.subheader('Choose the stock to fetch historical prices and charts')
import plotly.graph_objs as go
import plotly.figure_factory as ff
#@st.cache(persist=True)
start = datetime.datetime(2019, 1, 1)
end = datetime.datetime.today()
@st.cache(persist=True)
def micro_data():
    microsoft = pdr.get_data_yahoo('MSFT', start = start, end = end)
    return microsoft
@st.cache(persist=True)
def google_data():
    google = pdr.get_data_yahoo('GOOGL', start = start, end = end)
    return google
@st.cache(persist=True)
def apple_data():
    apple = pdr.get_data_yahoo('AAPL', start = start, end = end)
    return apple

microsoft = micro_data()
google = google_data()
apple = apple_data()
select = st.sidebar.radio('Stocks', ('Apple', 'Google', 'Microsoft','Compare'))

def companyDetails(company, companyName):
    st.write(f'## {companyName} Stock price over the past week')
    st.write(company.tail(7))
    st.write(f' {companyName} stock attributes from 2019 till today')

    fig = go.Figure(go.Scatter(x = company.index, y = company['High'], line=dict(color='#000072')))
    fig.update_yaxes(tickprefix="$")
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
#        st.line_chart(microsoft['High'])

if not st.sidebar.checkbox("Hide", False):
    if select == 'Microsoft':
        companyDetails(microsoft, "Microsoft")

    if select == 'Google':
        companyDetails(google, "Google")

    if select == 'Apple':
        companyDetails(apple, "Apple")

    if select == 'Compare':
        st.write('#### Assuming we invested $100 on these companies on first January of 2019, we would have this much money today')
        normalized_google = google.High.div(google.High.iloc[0]).mul(100)
        normalized_microsoft = microsoft.High.div(microsoft.High.iloc[0]).mul(100)
        normalized_apple = apple.High.div(apple.High.iloc[0]).mul(100)
        normalized_google.plot(figsize = (12,6))
        normalized_microsoft.plot()
        normalized_apple.plot()
        plt.legend(['Google','Microsoft','Apple'])
        plt.show()
        st.pyplot()

        st.write('#### Log returns on Histogram plot')
        stocks = pd.concat([microsoft.High,google.High,apple.High],axis=1)
        stocks.columns = ['Microsoft','Google','Apple']
        log_ret = np.log(stocks/stocks.shift(1))
        log_ret.hist(bins=60,figsize=(12,10));
        st.pyplot()

def candlestick_yearly(company, companyName):
    st.write('#### Yearly candlestick for ', companyName)
    trace = go.Candlestick(x=company['2020'].index,
    open=company['2020'].Open,
    high=company['2020'].High,
    low=company['2020'].Low,
    close=company['2020'].Close)
    data = [trace]
    #iplot(data, filename='simple_candlestick')
    st.plotly_chart(data)

def candlestick_monthly(company, companyName):
    st.write('#### Monthly candlestick for ', companyName)
    trace = go.Candlestick(x=company['11-2020'].index,
    open=company['11-2020'].Open,
    high=company['11-2020'].High,
    low=company['11-2020'].Low,
    close=company['11-2020'].Close)
    data = [trace]
    #iplot(data, filename='simple_candlestick')
    st.plotly_chart(data)


st.sidebar.subheader('Candlestick Chart on the basis of time')
if not st.sidebar.checkbox("Hide", True, key = '1'):
    duration = st.sidebar.selectbox('Duration', ('Monthly','Yearly'))

    if duration == 'Monthly':
        candlestick_monthly(microsoft, "Microsoft")
        candlestick_monthly(google, "Google")
        candlestick_monthly(apple, "Apple")

    if duration == 'Yearly':
        candlestick_yearly(microsoft, "Microsoft")
        candlestick_yearly(google, "Google")
        candlestick_yearly(apple, "Apple")


def ohlc_monthly(company, companyName):
    st.write('#### Monthly OHLC chart for ', companyName)
    trace = go.Ohlc(x=company['11-2020'].index,
            open=company['11-2020'].Open,
            high=company['11-2020'].High,
            low=company['11-2020'].Low,
            close=company['11-2020'].Close)
    data = [trace]
    st.plotly_chart(data)

def ohlc_yearly(company, companyName):
    st.write('#### Yearly OHLC chart for ', companyName)
    trace = go.Ohlc(x=company['2020'].index,
            open=company['2020'].Open,
            high=company['2020'].High,
            low=company['2020'].Low,
            close=company['2020'].Close)
    data = [trace]
    st.plotly_chart(data)

st.sidebar.subheader('OHLC Chart on the basis of time')
if not st.sidebar.checkbox("Hide", True, key = '2'):
    duration = st.sidebar.selectbox('Duration', ('Monthly','Yearly'))
    if duration == 'Monthly':
        ohlc_monthly(microsoft, "Microsoft")
        ohlc_monthly(google, "Google")
        ohlc_monthly(apple, "Apple")


    if duration == 'Yearly':
        ohlc_yearly(microsoft, "Microsoft")
        ohlc_yearly(google, "Google")
        ohlc_yearly(apple, "Apple")

def moving_avg_30(company, companyName):
    st.write('30 Day Moving Average for ', companyName)
    company['MA-30'] = company['Close'].rolling(30).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = company.index, y = company['MA-30'], name = "30 Day MA",line = dict(color='orange', width = 1.2)))
    fig.add_trace(go.Scatter(x = company.index, y = company['Close'], name = "Closing Price",line = dict(color='green', width = 1.2)))
    fig.update_xaxes(
        rangeslider_visible=True)

    st.write(fig)

def moving_avg_50(company, companyName):
    st.write('50 Day Moving Average for ', companyName)
    company['MA-50'] = company['Close'].rolling(50).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = company.index, y = company['MA-50'], name = "50 Day MA",line = dict(color='orange', width = 1.2)))
    fig.add_trace(go.Scatter(x = company.index, y = company['Close'], name = "Closing Price",line = dict(color='green', width = 1.2)))
    fig.update_xaxes(
        rangeslider_visible=True)

    st.write(fig)

def moving_avg_100(company, companyName):
    st.write('100 Day Moving Average for ', companyName)
    company['MA-100'] = company['Close'].rolling(100).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = company.index, y = company['MA-100'], name = "100 Day MA",line = dict(color='orange', width = 1.2)))
    fig.add_trace(go.Scatter(x = company.index, y = company['Close'], name = "Closing Price",line = dict(color='green', width = 1.2)))
    fig.update_xaxes(
        rangeslider_visible=True)

    st.write(fig)


st.sidebar.subheader('Moving Averages for these stocks')
if not st.sidebar.checkbox("Hide", True, key = '4'):
    duration = st.sidebar.selectbox('Duration', ('30MA','50MA','100MA'))
    if duration == '50MA':
        moving_avg_50(microsoft, "Microsoft")
        moving_avg_50(google, "Google")
        moving_avg_50(apple, "Apple")

    if duration == '100MA':
        moving_avg_100(microsoft, "Microsoft")
        moving_avg_100(google, "Google")
        moving_avg_100(apple, "Apple")

    if duration == '30MA':
        moving_avg_30(microsoft, "Microsoft")
        moving_avg_30(google, "Google")
        moving_avg_30(apple, "Apple")

def bollinger_bands(company, companyName):
    st.write('### Bollinger Bands')
    st.write('#### Bollinger Bands for ', companyName)
    company['20 day Close MA'] = company['Close'].rolling(20).mean()
    company['Upper'] = company['20 day Close MA'] + (2 * company['Close'].rolling(20).std())
    company['Lower'] = company['20 day Close MA'] - (2 * company['Close'].rolling(20).std())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = company.index, y = company['20 day Close MA'],
                  name = "20 Day MA",line = dict(color='rgb(117, 112, 179)', width = 1.2)))
    fig.add_trace(go.Scatter(x = company.index, y = company['Upper'],
                             name = "Upper Limit",line = dict(color='rgb(166, 86, 40)', width = 1.2)))
    fig.add_trace(go.Scatter(x = company.index, y = company['Lower'],
                             name = "Lower Limit",line = dict(color='#FD3216', width = 1.2)))
    fig.add_trace(go.Scatter(x = company.index, y = company['Close'],
                             name = "Closing Price",line = dict(color='rgb(27, 158, 119)', width = 1.2)))#'#19D3F3'
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
if not st.sidebar.checkbox("Hide", True, key = '5'):
    bollinger_bands(microsoft, "Microsoft")
    bollinger_bands(google, "Google")
    bollinger_bands(apple, "Apple")
