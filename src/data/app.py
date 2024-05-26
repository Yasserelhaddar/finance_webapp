# streamlit_app.py
import streamlit as st
from plotly import graph_objs as go
import pandas as pd
import datetime
import os
import time 
from PIL import Image

from utils import (extract_daily_data, extract_fx_daily_data, 
                   extract_commodity_data, extract_economic_indicators_api,
                   extract_technical_indicators_api, extract_news_sentiments, extract_filling_data, extract_intraday_data,
                   get_filing_text)

from langchain_utils import (chunk_text, create_vector_store, create_retriever, parse_query, generate_answer)

from langchain_core.messages import AIMessage, HumanMessage
import openai


def get_filling_extractor_input(filling_api_identity):

    filing_section_dict = {}

    CompanyTicker = st.sidebar.text_input("Company Ticker", "AAPL").upper()
    StartDate = st.date_input('Start Date', datetime.date(2022,1,1))
    EndDate = st.date_input('End Date', datetime.date(2023,1,1))
    FillingType = st.sidebar.selectbox("Filling Type", ["10-K", "10-Q", "8-K"])

    if FillingType == "10-K":
        FillingSection = st.sidebar.selectbox("Filling Section", ['Item 1', 'Item 1A', 'Item 1B', 'Item 2', 'Item 3', 'Item 5', 'Item 6', 'Item 7', 'Item 7A', 'Item 8', 'Item 9', 'Item 9A', 'Item 9B', 'Item 10', 'Item 11', 'Item 12', 'Item 13', 'Item 14', 'Item 15'])
    elif FillingType == "8-K":
        FillingSection = st.sidebar.selectbox("Filling Section", ['Item 1.01', 'Item 1.02', 'Item 1.03', 'Item 2.01', 'Item 2.02', 'Item 2.03', 'Item 2.04', 'Item 2.05', 'Item 2.06', 'Item 3.01', 'Item 3.02', 'Item 3.03', 'Item 4.01', 'Item 4.02', 'Item 5.01', 'Item 5.02', 'Item 5.03', 'Item 5.04', 'Item 5.05', 'Item 5.06', 'Item 5.0', 'Item 5.08', 'Item 6.01', 'Item 6.02', 'Item 6.03', 'Item 6.04', 'Item 6.05', 'Item 9.01'])
    elif FillingType == "10-Q":
        FillingSection = st.sidebar.selectbox("Filling Section", ['Item 1', 'Item 1A', 'Item 1B', 'Item 2', 'Item 3', 'Item 5', 'Item 6', 'Item 7', 'Item 7A', 'Item 8', 'Item 9', 'Item 9A', 'Item 9B', 'Item 10', 'Item 11', 'Item 12', 'Item 13', 'Item 14', 'Item 15'])

    extract_numeric = False

    filing_text, numeric_data = extract_filling_data(filling_api_identity, CompanyTicker, StartDate, EndDate, FillingType, FillingSection, extract_numeric)

    return filing_text

def get_vatange_api_input():
    data_type = st.sidebar.selectbox("Data Type", ["Select","Technical Indicators", "Commodities", 
                                                   "Economic Indicators in the US", "Daily Forex", 
                                                   "Daily Stock Market"])
    if data_type == "Technical Indicators":
        symbol = st.sidebar.text_input("Symbol", "AAPL").upper()
        price_type = st.sidebar.text_input("Data Type", "Close").upper()
        indicator = st.sidebar.selectbox("Indicator", ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 
                                                       'TRIMA', 'KAMA', 'MAMA', 'T3', 'MACD', 
                                                       'MACDEXT', 'STOCH', 'STOCHF', 'RSI', 
                                                       'STOCHRSI', 'WILLR', 'ADX', 'ADXR', 
                                                       'APO', 'PPO', 'MOM', 'BOP', 'CCI', 'CMO', 
                                                       'ROC', 'ROCR', 'AROON', 'AROONOSC', 'MFI', 
                                                       'TRIX', 'ULTOSC', 'DX', 'MINUS_DI', 
                                                       'PLUS_DI', 'MINUS_DM', 'PLUS_DM', 'BBANDS', 
                                                       'MIDPOINT', 'MIDPRICE', 'SAR', 'TRANGE', 
                                                       'ATR', 'NATR', 'AD', 'ADOSC', 'OBV', 
                                                       'HT_TRENDLINE', 'HT_SINE', 'HT_TRENDMODE', 
                                                       'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR'])
        return data_type, [symbol, indicator, price_type]

    elif data_type == "Daily Stock Market":
        symbol = st.sidebar.text_input("Symbol", "AAPL").upper()
        price_type = st.sidebar.selectbox("Price Type", ["Open", "High", "Low", "Close", "Adjusted Close"]).upper()
        return data_type, [symbol, price_type]

    elif data_type == "Commodities":
        commodity = st.sidebar.selectbox("Commodity", ['WTI', 'BRENT', 'NATURAL_GAS', 'COPPER', 'ALUMINUM', 'WHEAT', 'CORN', 'COTTON', 'SUGAR', 'COFFEE', 'ALL_COMMODITIES'])
        return data_type, [commodity]
        
    elif data_type == "Economic Indicators in the US":
        indicator = st.sidebar.selectbox("Indicator", ["INFLATION", "REAL_GDP", "REAL_GDP_PER_CAPITA", "CPI", "RETAIL_SALES", "DURABLES", "UNEMPLOYMENT", "NONFARM_PAYROLL", "TREASURY_YIELD", "FEDERAL_FUNDS_RATE"])
        return data_type, [indicator]
    
    elif data_type == "Daily Forex":
        currencies_df = pd.read_csv("/home/yasser/SM_project/src/data/physical_currency_list.csv")
        price_type = st.sidebar.selectbox("Price Type", ["Open", "High", "Low", "Close"]).upper()
        from_symbol = st.sidebar.selectbox("From", currencies_df['currency name'].to_list())
        to_symbol = st.sidebar.selectbox("To", currencies_df['currency name'].to_list())

        actual_from_symbol = str(currencies_df[currencies_df['currency name']==from_symbol]['currency code'].values[0])
        actual_to_symbol = str(currencies_df[currencies_df['currency name']==to_symbol]['currency code'].values[0])

        return data_type, [actual_from_symbol, actual_to_symbol, price_type]

    else:
        return None, []
        

# Function to decide which data fetching function to call based on user input
def fetch_statistic_data(api_key, data_type, param_list):
    if api_key and data_type != "Select":
        if data_type == "Technical Indicators":
            [symbol, indicator, price_type] = param_list
            if indicator != None:
                st.write(f"Fetching {data_type.lower()} daily stock data for symbol: {symbol}")
                data = extract_technical_indicators_api(indicator, symbol, "daily", 60, price_type.lower(), apikey=api_key)

        elif data_type == "Commodities":
            [commodity] = param_list
            
            if commodity in ['WTI', 'BRENT', 'NATURAL_GAS']:
                interval = "daily"
            else:
                interval = "monthly"
            data = extract_commodity_data(commodity, interval, apikey=api_key)  

        elif data_type == "Economic Indicators in the US":
            [indicator] = param_list
            data = extract_economic_indicators_api(indicator, apikey=api_key)  

        elif data_type == "Daily Forex":
            [actual_from_symbol, actual_to_symbol, price_type] = param_list
            data = extract_fx_daily_data(from_symbol=actual_from_symbol, to_symbol=actual_to_symbol, price_type=price_type.lower(), apikey=api_key) 

        elif data_type == "Daily Stock Market":
            [symbol, price_type] = param_list 
            data = extract_daily_data(symbol=symbol, price_type=price_type.lower(), apikey=api_key, adjusted=True)

        else:
            data = pd.DataFrame()
        
        if data is not None and not data.empty:
            return data

    return pd.DataFrame()


# Function to fetch intraday data
def fetch_intraday_data(symbol, interval, api_key):
    data = extract_intraday_data(symbol=symbol, interval=interval, apikey=api_key, adjusted="true", extended_hours="true", outputsize="full")

    if data.empty:
        st.error("Failed to fetch data. Please check your API key and symbol.")
        return pd.DataFrame()
    return data

# Function to plot intraday data
def plot_intraday_data(data):
    if not data.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['4. close'], mode='lines', name='Close Price'))
        fig.layout.update(title_text="Intraday Stock Price", xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data to display.")


# Plotting function
def plot_statistic_data(data, data_type, param_list):
    if data_type == "Technical Indicators":
        [symbol, indicator, price_type] = param_list
        plot_title = f"{indicator} on {price_type} price for {symbol}" 

    elif data_type == "Commodities":
        [commodity] = param_list
        plot_title = f"{commodity}" 
    
    elif data_type == "Economic Indicators in the US":
        [indicator] = param_list
        plot_title = f"{indicator}" 
    
    elif data_type == "Daily Forex":
        [actual_from_symbol, actual_to_symbol, price_type] = param_list
        plot_title = f"{price_type} Value from {actual_from_symbol} to {actual_to_symbol}" 

    elif data_type == "Daily Stock Market":
        [symbol, price_type] = param_list 
        plot_title = f"Daily {price_type} Stock Price for {symbol}" 

    if not data.empty:
        fig = go.Figure()
        # Assuming the 'y' values are in the first column for simplicity
        y_values = data.iloc[:, 0] if not data.empty else []
        fig.add_trace(go.Scatter(x=data.index, y=y_values, name=plot_title, mode='lines'))
        fig.layout.update(title_text=plot_title + " Over Time", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data to display.")

def display_news_item(row):
    # Display the news title with the link
    st.markdown(f"#### [{row['title']}]({row['url']})")

    col1, col2 = st.columns([4,1])

    with col1:
        # Display the news summary
        st.markdown(row['summary'])
    
    with col2:
        # Display the news image
        st.image(row['banner_image'], use_column_width=True)
        
    # Display sentiment and relevance scores with conditional color
    sentiment_color = "green" if float(row['ticker_ticker_sentiment_score']) > 0 else "red"
    relevance_color = "green" if float(row['ticker_relevance_score']) > 0.5 else "red"
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"<h4 style='color: {sentiment_color};'>Sentiment: {row['ticker_ticker_sentiment_label']}</h4>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<h4 style='color: {relevance_color};'>Relevance: {row['ticker_relevance_score']}</h4>", unsafe_allow_html=True)


def show_market_news_page(api_key):
    st.header("Market News Sentiment")
    
    # Assuming you have a predefined list of tickers or let the user input them
    selected_ticker = st.text_input("Enter ticker symbols separated by commas (e.g., IBM,MSFT,GOOG)", "")

    col1, col2 = st.columns(2)

    with col1:
        time_from = st.date_input('Start Date', datetime.date(2022,1,1))
    
    with col2:
        time_to = st.date_input('End Date', datetime.date(2023,1,1))


    if selected_ticker:
        # Fetch the news sentiment data
        news_df = extract_news_sentiments(apikey=api_key, function="NEWS_SENTIMENT", tickers=selected_ticker, topics=None, time_from=time_from.strftime("%Y%m%dT%H%M"), time_to=time_to.strftime("%Y%m%dT%H%M"), sort="RELEVANCE", limit=5)
        
        if not news_df.empty:
            # Filter news items for the selected ticker
            filtered_news_df = news_df[news_df['ticker_ticker'] == selected_ticker]
            
            if not filtered_news_df.empty:
                for _, row in filtered_news_df.iterrows():
                    if float(row['ticker_relevance_score']) > 0.7:
                        display_news_item(row)
            else:
                st.write(f"No news sentiment data found for {selected_ticker}")
        else:
            st.error("Failed to fetch news sentiment data. Please check your API key and symbol.")
    else:
        st.write("Please select a ticker symbol.")


def show_statistical_data_page(api_key):

    st.header("Statistical Data")

    # Sidebar for user inputs
    st.sidebar.header("User Input Parameters")


    data_type, param_list = get_vatange_api_input()
    if api_key:  # Ensures API key is provided before fetching data
        data = fetch_statistic_data(api_key, data_type, param_list)
        plot_statistic_data(data, data_type, param_list)
    else:
        st.warning("Please enter a valid Alpha Vantage API Key.")


def show_real_time_data_page(alpha_vantage_api_key):

    st.title("Real-Time Stock Data Viewer")

    # Sidebar for user inputs
    st.sidebar.header("User Input Parameters")
    symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL").upper()
    interval = st.sidebar.selectbox("Select Interval", ["1min", "5min", "15min", "30min", "60min"], index=0)
    api_key = st.sidebar.text_input("Enter Alpha Vantage API Key")

    trigger_fetch = st.sidebar.button("Start Auto-Fetch")
    if trigger_fetch:
        # Initialize or clear existing data
        st.session_state['data'] = pd.DataFrame()
        st.session_state['auto_fetch'] = True

    if 'auto_fetch' in st.session_state and st.session_state['auto_fetch']:
        # Fetch data and append
        new_data = fetch_intraday_data(symbol, interval, api_key)
        if 'data' not in st.session_state or st.session_state['data'].empty:
            st.session_state['data'] = new_data
        else:
            st.session_state['data'] = pd.concat([st.session_state['data'], new_data]).drop_duplicates()
        
        # Plot data
        plot_intraday_data(st.session_state['data'])
        time.sleep(60)  # Wait for 60 seconds before rerunning the app to update the plot
        st.experimental_rerun()

    st.sidebar.button("Stop Auto-Fetch", on_click=lambda: st.session_state.update({'auto_fetch': False}))

def show_filling_extractor_page(filling_api_identity):

    st.header("Filing Data")

    # Sidebar for user inputs
    st.sidebar.header("User Input Parameters")
    
    if filling_api_identity:  # Ensures API key is provided before fetching data
        filing_text = get_filling_extractor_input(filling_api_identity)

        st.markdown(filing_text)

    else:
        st.warning("Please enter a valid EDGAR Identity.")
    
def show_filling_chatbot_page(filing_api_identity, openai_api_key):

    st.header("Filing ChatBot")
    query = st.text_input("Ask me something about company filings:")

    if query:
        # Parse query using an LLM to extract details
        print("Parsing Query")
        ticker, year, filing_type, section = parse_query(query, openai_api_key)
        print(ticker, year, filing_type, section)
        # Retrieve filing text
        print("Getting Filling data")
        filing_path, filing_text_exists = get_filing_text(filing_api_identity ,ticker, year, filing_type, section)

        if filing_text_exists:
            # Chunk, store, and retrieve relevant context
            print("Chunking Filling data")
            document_chunks = chunk_text(filing_path)
            print("number of chunks: ", len(document_chunks))
            print("Creating Vector Store")
            vector_store = create_vector_store(document_chunks, openai_api_key)
            print("Creating Retriever")
            retriever_chain = create_retriever(vector_store, openai_api_key)
            
            # Retrieve the most relevant document chunk
            print("Retrieving relevant document")
            context = retriever_chain.invoke({"chat_history": st.session_state.chat_history, "input": query})
            

            # Generate response using the context
            response = generate_answer(context, query, openai_api_key)
            st.write(response)
            st.session_state.chat_history.append(HumanMessage(content=query))
            st.session_state.chat_history.append(AIMessage(content=response))


# Set page config
st.set_page_config(page_title="Finance WebApp", layout="wide")

# Streamlit app title
st.title("Finance WebApp")
st.sidebar.header("Navigation")


options = ["Select", "Alpha Vantage - Data Visualization", "Alpha Vantage - Market News Sentiment", "Real Time Platform", "Filling Extractor", "Filling ChatBot"]
selection = st.sidebar.selectbox("Choose a page", options)

# Page router
if selection == "Alpha Vantage - Data Visualization":
    # Sidebar for navigation
    api_key = st.sidebar.text_input("API Key", "")
    show_statistical_data_page(api_key)

elif selection == "Alpha Vantage - Market News Sentiment":
    api_key = st.sidebar.text_input("API Key", "")
    show_market_news_page(api_key)

elif selection == "Real Time Platform":
    alpha_vantage_api_key = st.sidebar.text_input("Alpha Vantage API Key", "")
    show_real_time_data_page(alpha_vantage_api_key)

elif selection == "Filling Extractor":
    filling_api_identity = st.sidebar.text_input("EDGAR API Idenitity: {FirstName} {LastName} {Email}")
    show_filling_extractor_page(filling_api_identity)

elif selection == "Filling ChatBot":
    # Ensure that chat_history is initialized in session_state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    openai_api_key = st.sidebar.text_input("OPEN AI API", "")
    openai.api_key = openai_api_key
    filling_api_identity = st.sidebar.text_input("EDGAR API Idenitity", "")
    show_filling_chatbot_page(filling_api_identity, openai_api_key)


else:
    # Load and display an image
    image_path = '/home/yasser/SM_project/assets/image.webp'  # Update this to your image path
    image = Image.open(image_path)
    st.image(image, caption='Welcome to Finance WebApp', use_column_width=True)
    st.write("Select a page from the dropdown to get started.")



