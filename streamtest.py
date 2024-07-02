
import streamlit as st
import yfinance as yf
import pandas as pd
import QuantLib as ql
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import cufflinks as cf
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.dates as mdates

# Sidebar
st.sidebar.title("Choose the stock parameters:")
st.sidebar.write('Choose the start and end date for the stock prices to yield data. Closing Price of stock in end date will be used for option pricing.')
start_date = st.sidebar.date_input("Start date", date(2024, 1, 1))
end_date = st.sidebar.date_input("End date", date(2024, 1, 31))
st.sidebar.write("Choose the Ticker in NSE:")
tickers = pd.read_html('https://ournifty.com/stock-list-in-nse-fo-futures-and-options.html#:~:text=NSE%20F%26O%20Stock%20List%3A%20%20%20%20SL,%20%201000%20%2052%20more%20rows%20')[0]
tickers = tickers.SYMBOL.to_list()
for count in range(len(tickers)):
    tickers[count] = tickers[count] + ".NS"
tickers = tickers[4:]
ticker = st.sidebar.selectbox("Ticker", tickers)

#OPtion Parameters
st.sidebar.title("Choose the option parameters:")
strike_price = st.sidebar.number_input("Strike Price", value=1500)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=0.06)
expiration_date = st.sidebar.date_input("Expiration Date", date(2024, 7, 1))
option_type = st.sidebar.radio("Option Type", ["Call", "Put"])
option_type = ql.Option.Call if option_type == "Call" else ql.Option.Put




#------------------------ Main content ---------------------------------------------------
st.markdown('''
# Option Pricing and Greeks Calculator using Black-Scholes Model
Below is the details of the Stock you chose! Choose the option parameters to yield the option prices and Greeks.
            
**Credits**
- App built by Kaushik Srivatsan
- Built in `Python` using `streamlit`,`yfinance`, `matplotlib`, `pandas`, `plotly` and `datetime`
''')
st.write('---')
# Getting info from yfinance 
tickerData = yf.Ticker(ticker) # Get ticker data
# Get the historical prices for this ticker period determined by the user
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date) 

# Ticker information - Extracting the name of the stock.
string_name = tickerData.info['longName']
st.header('**%s**' % string_name)

def ticker_info(tickerData, tickerdf):
    string_summary = tickerData.info['longBusinessSummary']
    st.write("***About the Company***")
    st.write(string_summary)

    # Ticker data
    st.header('**Ticker data**')
    st.write(tickerDf)

def fetch_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Returns'] = stock_data['Close'].pct_change()
    stock_data['Returns'] = stock_data['Returns'].fillna(0)
    closing_price = stock_data['Close'][-1]
    return stock_data, closing_price

def create_stock_diagram(stock_data):
    # Extract the dates and closing prices from the stock_data DataFrame
    dates = stock_data.index
    closing_prices = stock_data['Close']

    # Creating the Plotly figure
    fig = px.line(stock_data, x=stock_data.index, y='Close', 
                title='Stock Price Chart',
                labels={'Close': 'Price', 'index': 'Date'},
                template='plotly_white')

    # Customize the layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        legend_title_text='',
        xaxis_rangeslider_visible=True
    )

    # Add range selector
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

def handle_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None
    return wrapper

def validate_inputs(start_date, end_date, ticker, strike_price, risk_free_rate):
    if start_date >= end_date:
        st.warning("End date must be after start date.")
        return False
    # only choose days when the market is open
    if start_date.weekday() > 4 or end_date.weekday() > 4:
        st.warning("Please choose a weekday.")
        return False
    if ticker == "":
        st.warning("Please enter a valid stock ticker.")
        return False
    if strike_price <= 0:
        st.warning("Strike price must be positive.")
        return False
    if risk_free_rate < 0 or risk_free_rate > 1:
        st.warning("Risk-free rate should be between 0 and 1.")
        return False
    return True

def validate_sigma(sigma):
    if sigma <= 0 or sigma > 2:
        st.warning("Volatility should be positive and typically less than 2.")
        return False
    return True

def compute_historical_volatility(stock_data):
    stock_data['Returns'] = stock_data['Close'].pct_change()
    daily_volatility = stock_data['Returns'].std()
    annual_volatility = daily_volatility * np.sqrt(360)  # 252 trading days in a year
    return annual_volatility

def create_option_payoff_diagram(S, K, option_price, option_type):
    # Create a range of stock prices from 0.5 to 1.5 times the current stock price
    stock_prices = np.linspace(0.5 * S, 1.5 * S, 200)
    
    # Calculate payoffs
    if option_type == ql.Option.Call:
        payoffs = np.maximum(stock_prices - K, 0) - option_price
    else:  # Put option
        payoffs = np.maximum(K - stock_prices, 0) - option_price
    
    # Calculate break-even point
    break_even = K + option_price if option_type == ql.Option.Call else K - option_price
    
    # Create the figure
    fig = go.Figure()
    
    # Add option payoff line
    fig.add_trace(go.Scatter(x=stock_prices, y=payoffs, mode='lines', name='Option Payoff'))
    
    # Add current stock price line
    fig.add_trace(go.Scatter(x=[S, S], y=[min(payoffs), max(payoffs)], mode='lines', name='Current Stock Price', line=dict(color='red', dash='dash')))
    
    # Add strike price line
    fig.add_trace(go.Scatter(x=[K, K], y=[min(payoffs), max(payoffs)], mode='lines', name='Strike Price', line=dict(color='green', dash='dash')))
    
    # Add break-even line
    fig.add_trace(go.Scatter(x=[break_even, break_even], y=[min(payoffs), max(payoffs)], mode='lines', name='Break-Even', line=dict(color='orange', dash='dot')))

    
    # Update layout
    fig.update_layout(
        title=f'{"Call" if option_type == ql.Option.Call else "Put"} Option Payoff Diagram',
        xaxis_title='Stock Price',
        yaxis_title='Profit/Loss',
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99),
        template='plotly_white'
    )
    fig.add_shape(type="line", x0=S, y0=min(payoffs), x1=S, y1=max(payoffs), line=dict(color="red", width=1, dash="dash"))
 
    fig.update_yaxes(zeroline=True, zerolinecolor='gray', zerolinewidth=1)

    return fig

# Call option prices and Greeks
@handle_error
def calculate_option_prices_and_greeks(S, K, T, r, sigma, option_type):
    # Option parameters
    payoff = ql.PlainVanillaPayoff(option_type, K)
    exercise = ql.EuropeanExercise(T)
    
    # Market data
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(r)), ql.Actual360()))
    vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(sigma)), ql.Actual360()))
    
    # Black-Scholes process
    bsm_process = ql.BlackScholesMertonProcess(spot_handle, flat_ts, flat_ts, vol_handle)
    
    # European option
    option = ql.EuropeanOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))

    # Option price
    call_price = option.NPV()
    
    # Greeks
    delta = option.delta()
    gamma = option.gamma()
    vega = option.vega()
    theta = option.theta()
    
    return call_price, delta, gamma, vega, theta


T = ql.Date(expiration_date.day, expiration_date.month, expiration_date.year)
if validate_inputs(start_date, end_date, ticker, strike_price, risk_free_rate):
    stock_data, closing_price = fetch_data(ticker, start_date, end_date)
    create_stock_diagram(stock_data)
    ticker_info(tickerData, tickerDf)
    S = stock_data['Close'][-1]
    sigma = compute_historical_volatility(stock_data)
    if validate_sigma(sigma):
        call_price, call_delta, call_gamma, call_vega, call_theta = calculate_option_prices_and_greeks(S, strike_price, T, risk_free_rate, sigma, option_type)

    if call_price is not None:
        #Display in a neat streamlit Coloumn table format
        st.subheader("Option Prices and Greeks")
        option_type = "Call" if option_type == ql.Option.Call else "Put"
        option_data = {
            "Option Type": [option_type],
            "Strike Price": [strike_price],
            "Risk-Free Rate": [risk_free_rate],
            "Expiration Date": [expiration_date],
            "Stock Price": [S],
            "Option Price": [call_price],
            "Delta": [call_delta],
            "Gamma": [call_gamma],
            "Vega": [call_vega],
            "Theta": [call_theta]
        }
        option_df = pd.DataFrame(option_data)
        st.write(option_df)

         # Display option payoff diagram
        payoff_diagram = create_option_payoff_diagram(S, strike_price, call_price, option_type)
        st.plotly_chart(payoff_diagram, use_container_width=True)





# Display information on Black scholes model and equations
st.subheader('**Black-Scholes Model**')
st.markdown('''
The Black-Scholes model, also known as the Black-Scholes-Merton (BSM) model, is one of the most important concepts in modern financial theory. 
This mathematical equation estimates the theoretical value of derivatives based on other investment instruments, taking into account the impact of time and other risk factors.

**Black-Scholes Assumptions**
The Black-Scholes model makes certain assumptions:

- No dividends are paid out during the life of the option.
- Markets are random (i.e., market movements cannot be predicted).
- There are no transaction costs in buying the option.
- The risk-free rate and volatility of the underlying asset are known and constant.
- The returns of the underlying asset are normally distributed.
- The option is European and can only be exercised at expiration.
          
**Black-Scholes Formula** ''')
st.latex(r'''
    C(S, t) = S_t N(d_1) - K e^{-r(T-t)} N(d_2)
''')
st.latex(r'''
    d_1 = \frac{ln(S_t/K) + (r + \frac{\sigma^2}{2})(T-t)}{\sigma \sqrt{T-t}}
''')
st.latex(r'''
    d_2 = d_1 - \sigma \sqrt{T-t}
''')
st.latex(r'''
    \Delta = N(d_1)
''')
st.latex(r'''
    \Gamma = \frac{N'(d_1)}{S_t \sigma \sqrt{T-t}}
''')
st.latex(r'''
    \Theta = -\frac{S_t N'(d_1) \sigma}{2 \sqrt{T-t}} - r K e^{-r(T-t)} N(d_2)
''')
st.latex(r'''
    Vega = S_t N'(d_1) \sqrt{T-t}
''')

st.markdown('''
Where:
- C(S, t) is the call option price
- S is the stock price
- K is the strike price
- r is the risk-free rate
- T is the time to expiration
- t is the current time
- N is the cumulative distribution function of the standard normal distribution
- N' is the probability density function of the standard normal distribution
''')

