import requests
import csv
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from edgar import set_identity, Company

import os 
import datetime


##### UTILS #######

def process_json(json_data, function):
    """
    Process the API Output JSON data into a DataFrame.
    
    Parameters:
    - json_data: The JSON data from the API response.
    - function: The API function used, affecting the data structure (e.g., "SMA", "TIME_SERIES_INTRADAY").
    
    Returns:
    - DataFrame of the processed data.
    """
    
    # Extracting the main data section from JSON
    if function in ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'MAMA', 'VWAP', 'T3', 'MACD', 'MACDEXT', 'STOCH', 'STOCHF', 'RSI', 'STOCHRSI', 'WILLR', 'ADX', 'ADXR', 'APO', 'PPO', 'MOM', 'BOP', 'CCI', 'CMO', 'ROC', 'ROCR', 'AROON', 'AROONOSC', 'MFI', 'TRIX', 'ULTOSC', 'DX', 'MINUS_DI', 'PLUS_DI', 'MINUS_DM', 'PLUS_DM', 'BBANDS', 'MIDPOINT', 'MIDPRICE', 'SAR', 'TRANGE', 'ATR', 'NATR', 'AD', 'ADOSC', 'OBV', 'HT_TRENDLINE', 'HT_SINE', 'HT_TRENDMODE', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR']:
        data_section = list(json_data.values())[1]  # Technical indicators have their data in the second key
    elif function in ['TIME_SERIES_INTRADAY', 'TIME_SERIES_DAILY', 'TIME_SERIES_DAILY_ADJUSTED', 'FX_INTRADAY', 'FX_DAILY']:
        # For time series data, the key naming varies, so we need to adapt dynamically
        data_section = json_data[next(key for key in json_data.keys() if 'Time Series' in key or 'Technical Analysis' in key)]
    elif function in ["MARKET_STATUS"]:
        data_section = json_data['markets']
    elif function in ["INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW", "EARNINGS"]:
        key_map = {
            "INCOME_STATEMENT": "annualReports",
            "BALANCE_SHEET": "annualReports",
            "CASH_FLOW": "annualReports",
            "EARNINGS": "annualEarnings"
        }
        data_section = json_data[key_map[function]]
    elif function in ["NEWS_SENTIMENT"]:
        data_section = json_data['feed']
    elif function in ['WTI', 'BRENT', 'NATURAL_GAS', 'COPPER', 'ALUMINUM', 'WHEAT', 'CORN', 'COTTON', 'SUGAR', 'COFFEE', 'ALL_COMMODITIES', "REAL_GDP", "REAL_GDP_PER_CAPITA", "TREASURY_YIELD", "FEDERAL_FUNDS_RATE", "CPI", "INFLATION", "RETAIL_SALES", "DURABLES", "UNEMPLOYMENT", "NONFARM_PAYROLL"]:
        data_section = json_data['data']
    else:
        data_section = json_data


    # Convert the retrieved values to a DataFrame
    if function in ['TIME_SERIES_INTRADAY', 'TIME_SERIES_DAILY', 'TIME_SERIES_DAILY_ADJUSTED', 'FX_INTRADAY', 'FX_DAILY']:
        # For time series, we need to process each time entry into columns
        data_df = pd.DataFrame.from_dict(data_section, orient='index').astype(float)
        data_df.index = pd.to_datetime(data_df.index)
    elif function in ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'MAMA', 'VWAP', 'T3', 'MACD', 'MACDEXT', 'STOCH', 'STOCHF', 'RSI', 'STOCHRSI', 'WILLR', 'ADX', 'ADXR', 'APO', 'PPO', 'MOM', 'BOP', 'CCI', 'CMO', 'ROC', 'ROCR', 'AROON', 'AROONOSC', 'MFI', 'TRIX', 'ULTOSC', 'DX', 'MINUS_DI', 'PLUS_DI', 'MINUS_DM', 'PLUS_DM', 'BBANDS', 'MIDPOINT', 'MIDPRICE', 'SAR', 'TRANGE', 'ATR', 'NATR', 'AD', 'ADOSC', 'OBV', 'HT_TRENDLINE', 'HT_SINE', 'HT_TRENDMODE', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR']:
        # For technical indicators, the structure is simpler
        data_df = pd.DataFrame.from_dict(data_section, orient='index')
        data_df.index = pd.to_datetime(data_df.index)
        # Convert all columns to numeric, errors='coerce' will set invalid parsing to NaN
        data_df = data_df.apply(pd.to_numeric, errors='coerce')
    elif function in ["OVERVIEW", "MARKET_STATUS", "INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW", "EARNINGS"]:
        if function == "OVERVIEW":
            data_df = pd.DataFrame(data_section, index=[0])
        else:
            data_df = pd.DataFrame(data_section)
    elif function in ["NEWS_SENTIMENT"]:
        data_df = pd.json_normalize(data_section, 'ticker_sentiment', 
                            meta=['title', 'url', 'time_published', 'authors', 'summary', 
                                    'banner_image', 'source', 'category_within_source', 
                                    'source_domain', 'topics', 'overall_sentiment_score', 
                                    'overall_sentiment_label'],
                            record_prefix='ticker_')

        # Post-processing to handle authors and topics as needed
        # For example, if you want to join the authors list into a single string
        data_df['authors'] = data_df['authors'].apply(lambda x: ', '.join(x))

        # Assuming topics is a list of dictionaries where you want to join all 'topic' values into a string
        data_df['topics'] = data_df['topics'].apply(lambda x: ', '.join([d['topic'] for d in x]))
    elif function in ['WTI', 'BRENT', 'NATURAL_GAS', 'COPPER', 'ALUMINUM', 'WHEAT', 'CORN', 'COTTON', 'SUGAR', 'COFFEE', 'ALL_COMMODITIES', "REAL_GDP", "REAL_GDP_PER_CAPITA", "TREASURY_YIELD", "FEDERAL_FUNDS_RATE", "CPI", "INFLATION", "RETAIL_SALES", "DURABLES", "UNEMPLOYMENT", "NONFARM_PAYROLL"]:
        data_df = pd.DataFrame(data_section)

        # if function in ['COPPER', 'ALUMINUM', 'WHEAT', 'CORN', 'COTTON', 'SUGAR', 'COFFEE', 'ALL_COMMODITIES', "TREASURY_YIELD", "FEDERAL_FUNDS_RATE", "CPI", "RETAIL_SALES", "DURABLES", "UNEMPLOYMENT", "NONFARM_PAYROLL"]:

            # print(data_df)
            # value = data_df[data_df['date']=="2023-12-01"]['value'][0]

            # new_rows = pd.DataFrame({
            # "date": ["2024-01-02"],
            # "value": [value]  
            # })

            # data_df = pd.concat([data_df, new_rows])
            # data_df = data_df.sort_index(ascending=False)

        if function in ["INFLATION"]:
            new_rows = pd.DataFrame({
            "date": ["2024-01-01", "2023-01-01"],
            "value": [3.1, 4.1]  
            })

            data_df = pd.concat([data_df, new_rows])
            data_df = data_df.sort_index(ascending=False)

        if function in ["REAL_GDP"]:
            new_rows = pd.DataFrame({
            "date": ["2023-10-02", "2024-01-02"],
            "value": [5642.697, 5642.697]  
            })

            data_df = pd.concat([data_df, new_rows])
            data_df = data_df.sort_index(ascending=False)

        if function in ["REAL_GDP_PER_CAPITA"]:
            new_rows = pd.DataFrame({
            "date": ["2023-10-02", "2024-01-02"],
            "value": [67050.0,  67050.0]  
            })

            data_df = pd.concat([data_df, new_rows])
            data_df = data_df.sort_index(ascending=False)

        data_df['value'] = pd.to_numeric(data_df['value'], errors='coerce')  # Convert 'value' to numeric, handling any errors
        # Convert the 'date' column to datetime format
        data_df['date'] = pd.to_datetime(data_df['date'])
        # Set the 'date' column as the index of the DataFrame
        data_df = data_df.set_index('date')
    else:
        pass

    return data_df


def fetch_data(request_endpoint, function, datatype="json"):
    """
    Helper function to fetch data from the Alpha Vantage API.
    
    Parameters:
    - request_endpoint: The full request URL.
    - datatype: Format of the returned data ("json" or "csv").
    
    Returns:
    - Data from the Alpha Vantage API as JSON or CSV.
    """
    try:
        response = requests.get(request_endpoint)
        response.raise_for_status()
        if datatype == "json":
            data_df = process_json(response.json(), function)
        else:
            data_df = process_json(response.text, function)

        return data_df 
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
    
    return None

def plot_time_series(dataframe, indicator_col):
    """
    Plots the time series and Technical Indicator values from the provided DataFrame using Plotly.
    
    Parameters:
    - dataframe: A pandas DataFrame with a datetime index and a column for the Technical Indicator to plot.
    - indicator_col: The name of the column representing the Technical Indicator to plot.
    """
    # Create a Plotly figure
    fig = go.Figure()

    # Add the time series data
    fig.add_trace(go.Scatter(x=dataframe.index, y=dataframe[indicator_col], mode='lines', name=indicator_col))

    # Update layout with title and axis labels
    fig.update_layout(title=f'{indicator_col} Over Time',
                      xaxis_title='Time',
                      yaxis_title='Value',
                      xaxis=dict(
                          tickmode='auto',  # Plotly handles the tick mode automatically
                          nticks=20,  # Suggests to Plotly how many ticks to display, but Plotly will adjust as needed
                          tickangle=45  # Rotate date labels for better readability
                      ),
                      yaxis=dict(
                          tickmode='auto',  # Allows Plotly to manage y-axis ticks automatically
                      ))

    # Display the plot
    fig.show()


def merge_data_frames(base_df, *additional_dfs):
    """
    Merges additional DataFrames into the base DataFrame. The function assumes
    that all DataFrames have datetime indices and that the base_df has the highest frequency.
    
    Parameters:
    - base_df: The base DataFrame with the highest frequency (daily data).
    - additional_dfs: Tuples containing the additional DataFrames to merge and their frequencies.
                      Each tuple is in the format (DataFrame, 'D', 'M', 'Q', or 'A').
    
    Returns:
    - Merged DataFrame with duplicated values for lower-frequency data to match the base DataFrame's frequency.
    """
    merged_df = base_df.copy()
    for df, freq in additional_dfs:
        if freq != 'D':  # If not daily, resample and forward fill
            df = df.resample(freq).ffill().reindex(merged_df.index, method='ffill')
        merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, how='left')
    return merged_df

##### TECHNICAL INDICATORS APIs #######

def extract_economic_indicators_api(function, datatype="json", apikey=""):
    """
    Extracts technical indicators from the Alpha Vantage API.
    
    Parameters:
    - function: The economic indicator function (e.g., "REAL_GDP").
    - interval: Time interval between data points ("daily", "quarterly", "annual", etc.).
    - datatype: Format of the returned data ("json" or "csv").
    - apikey: Your Alpha Vantage API key.
    
    Returns:
    - Data from the Alpha Vantage API as JSON or CSV.
    """
    if function in ["TREASURY_YIELD", "FEDERAL_FUNDS_RATE"]:
        interval = "daily"
    elif function == "CPI":
        interval = "monthly"
    elif function == "REAL_GDP":
        interval = "quarterly"
    else:
        interval = None

    # ["Q-REAL_GDP", "Nan-A-REAL_GDP_PER_CAPITA", "D-TREASURY_YIELD", "D-FEDERAL_FUNDS_RATE", "M-CPI", "Nan-A-INFLATION", "Nan-M-RETAIL_SALES", "Nan-M-DURABLES", "Nan-M-UNEMPLOYMENT", "Nan-M-NONFARM_PAYROLL"]
    # Constructing the request URL
    request_endpoint = f"https://www.alphavantage.co/query?function={function}&datatype={datatype}&apikey={apikey}"
    
    # Adding the 'month' parameter if provided
    if interval:
        request_endpoint += f"&interval={interval}"
    economic_indicator_df = fetch_data(request_endpoint, function, datatype="json")

    return economic_indicator_df


##### TECHNICAL INDICATORS APIs #######

def extract_technical_indicators_api(function, symbol, interval, time_period, series_type, datatype="json", apikey="", month=None):
    """
    Extracts technical indicators from the Alpha Vantage API.
    
    Parameters:
    - function: The technical indicator function (e.g., "SMA").
    - symbol: The ticker symbol (e.g., "IBM").
    - interval: Time interval between data points ("1min", "5min", etc.).
    - time_period: Number of data points for moving average.
    - series_type: Type of price in time series ("close", "open", etc.).
    - datatype: Format of the returned data ("json" or "csv").
    - apikey: Your Alpha Vantage API key.
    - month: Optional, specify month for intraday intervals in "YYYY-MM" format.
    
    Returns:
    - Data from the Alpha Vantage API as JSON or CSV.
    """
    
    # Constructing the request URL
    request_endpoint = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&datatype={datatype}&apikey={apikey}"
    
    # Adding the 'month' parameter if provided
    if month:
        request_endpoint += f"&month={month}"
    tech_indicator_df = fetch_data(request_endpoint, function, datatype="json")

    return tech_indicator_df


##### CORE STOCK APIs #######

def fetch_market_status(apikey, function="MARKET_STATUS"):
    """
    Fetches the current market status (open vs. closed) of major global trading venues.
    
    Parameters:
    - apikey: Your Alpha Vantage API key.
    
    Returns:
    - The market status data from the Alpha Vantage API as JSON.
    """
    
    # Constructing the request URL
    request_endpoint = f"https://www.alphavantage.co/query?function={function}&apikey={apikey}"

    market_status_df = fetch_data(request_endpoint, function, datatype="json")
    
    return market_status_df


def extract_intraday_data(symbol, interval, apikey, function="TIME_SERIES_INTRADAY", adjusted="true", extended_hours="true", month=None, outputsize="compact", datatype="json"):
    """
    Extracts intraday OHLCV time series data from the Alpha Vantage API.
    
    Parameters:
    - function: The time series function, default is "TIME_SERIES_INTRADAY".
    - symbol: The ticker symbol (e.g., "IBM").
    - interval: Time interval between two consecutive data points ("1min", "5min", etc.).
    - adjusted: Set to "true" for split/dividend-adjusted data, "false" for raw data.
    - extended_hours: Include extended trading hours if "true", else only regular hours if "false".
    - month: Optional, specify month for intraday intervals in "YYYY-MM" format.
    - outputsize: "compact" for the latest 100 data points, "full" for more comprehensive data.
    - datatype: Format of the returned data ("json" or "csv").
    - apikey: Your Alpha Vantage API key.
    
    Returns:
    - Data from the Alpha Vantage API as JSON or CSV.
    """
    
    # Constructing the request URL
    request_endpoint = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&apikey={apikey}&adjusted={adjusted}&extended_hours={extended_hours}&outputsize={outputsize}&datatype={datatype}"
    
    # Adding the 'month' parameter if provided
    if month:
        request_endpoint += f"&month={month}"
    
    intraday_data_df = fetch_data(request_endpoint, function, datatype="json")
    
    return intraday_data_df
    


def extract_daily_data(symbol, price_type, apikey, adjusted=True, outputsize="compact", datatype="json"):
    """
    Extracts daily or daily adjusted time series data from the Alpha Vantage API.
    
    Parameters:
    - symbol: The ticker symbol (e.g., "IBM").
    - apikey: Your Alpha Vantage API key.
    - adjusted: Boolean, set to True for daily adjusted time series, False for raw daily time series.
    - outputsize: "compact" for the latest 100 data points, "full" for the full-length time series.
    - datatype: Format of the returned data ("json" or "csv").
    
    Returns:
    - Data from the Alpha Vantage API as JSON or CSV.
    """
    dict_price_types = {"open": "1. open", "high": "2. high", "low": "3. low", "close": "4. close", "adjusted close": "5. adjusted close"}

    function = "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY"
    
    # Constructing the request URL
    request_endpoint = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={apikey}&outputsize={outputsize}&datatype={datatype}"
    
    daily_data_df = fetch_data(request_endpoint, function, datatype="json")

    df_subset = daily_data_df[[dict_price_types[price_type]]]
    df_subset = df_subset.rename(columns={dict_price_types[price_type]: f'{price_type.upper()}'})

    return df_subset

##### FUNDAMENTAL DATA APIs #######

def extract_fundamental_data(function, symbol, apikey, date=None, state=None, horizon=None):
    """
    Extracts various types of financial data from the Alpha Vantage API, including income statements,
    balance sheets, cash flows, listing status, earnings calendar, and IPO calendar.
    
    Parameters:
    - function: The API function name.
    - symbol: The ticker symbol (e.g., "IBM"). Optional for some functions.
    - apikey: Your Alpha Vantage API key.
    - date: Specific date for listing status, optional.
    - state: "active" or "delisted" for listing status, optional.
    - horizon: "3month", "6month", or "12month" for earnings calendar, optional.
    
    Returns:
    - Data as a pandas DataFrame.
    """
    # Construct request endpoint
    request_endpoint = f"https://www.alphavantage.co/query?function={function}&apikey={apikey}"
    
    if symbol:
        request_endpoint += f"&symbol={symbol}"
    if date:
        request_endpoint += f"&date={date}"
    if state:
        request_endpoint += f"&state={state}"
    if horizon:
        request_endpoint += f"&horizon={horizon}"
    if function in ["INCOME_STATEMENT", "INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW", "EARNINGS", "OVERVIEW"]:
        return fetch_data(request_endpoint, function, datatype="json")
    else:
        with requests.Session() as s:
            download = s.get(request_endpoint)
            decoded_content = download.content.decode('utf-8')
            
            # Convert CSV string to DataFrame
            data = StringIO(decoded_content)
            df = pd.read_csv(data, sep=",")
            
            return df

##### MARKET NEWS AND SENTIMENTS APIs #######

def extract_news_sentiments(apikey, function="NEWS_SENTIMENT", tickers=None, topics=None, time_from="19990101T0000", time_to="19990102T0000", sort="RELEVANCE", limit=1):
    """
    Fetches market news and sentiment data from the Alpha Vantage API.
    
    Parameters:
    - apikey: Your Alpha Vantage API key.
    - function: The API function name, default is "NEWS_SENTIMENT".
    - tickers: Comma-separated list of stock/crypto/forex symbols (optional).
    - topics: Comma-separated list of news topics (optional).
    - time_from: Start time for news articles, in YYYYMMDDTHHMM format (optional).
    - time_to: End time for news articles, in YYYYMMDDTHHMM format (optional).
    - sort: Sort order of the articles ("LATEST", "EARLIEST", "RELEVANCE") (optional).
    - limit: Maximum number of results to return (optional).
    
    Returns:
    - Data as a pandas DataFrame.
    """
    # Construct request endpoint
    request_endpoint = f"https://www.alphavantage.co/query?function={function}&apikey={apikey}&sort={sort}&limit={limit}"
    
    if tickers:
        request_endpoint += f"&tickers={tickers}"
    if topics:
        request_endpoint += f"&topics={topics}"
    if time_from:
        request_endpoint += f"&time_from={time_from}"
    if time_to:
        request_endpoint += f"&time_to={time_to}"
    
    # Fetching the data
    return fetch_data(request_endpoint, function, datatype="json")

##### FOREX APIs #######

def extract_fx_intraday_data(from_symbol, to_symbol, interval, apikey, outputsize="compact", datatype="json"):
    """
    Extracts intraday Forex time series data from the Alpha Vantage API.
    
    Parameters:
    - from_symbol: The three-letter symbol for the base currency (e.g., "EUR").
    - to_symbol: The three-letter symbol for the quote currency (e.g., "USD").
    - interval: Time interval between two consecutive data points ("1min", "5min", etc.).
    - outputsize: "compact" for the latest 100 data points, "full" for more comprehensive data.
    - datatype: Format of the returned data ("json" or "csv").
    - apikey: Your Alpha Vantage API key.
    
    Returns:
    - Data from the Alpha Vantage API as JSON or CSV.
    """
    
    function = "FX_INTRADAY"
    request_endpoint = f"https://www.alphavantage.co/query?function={function}&from_symbol={from_symbol}&to_symbol={to_symbol}&interval={interval}&apikey={apikey}&outputsize={outputsize}&datatype={datatype}"
    
    intraday_fx_data_df = fetch_data(request_endpoint, function, datatype)
    
    return intraday_fx_data_df


def extract_fx_daily_data(from_symbol, to_symbol, price_type, apikey, outputsize="compact", datatype="json"):
    """
    Extracts daily Forex time series data from the Alpha Vantage API.
    
    Parameters:
    - from_symbol: The three-letter symbol for the base currency (e.g., "EUR").
    - to_symbol: The three-letter symbol for the quote currency (e.g., "USD").
    - outputsize: "compact" for the latest 100 data points, "full" for more comprehensive data.
    - datatype: Format of the returned data ("json" or "csv").
    - apikey: Your Alpha Vantage API key.
    
    Returns:
    - Data from the Alpha Vantage API as JSON or CSV.
    """
    dict_price_types = {"open": "1. open", "high": "2. high", "low": "3. low", "close": "4. close", "adjusted close": "5. adjusted close"}

    function = "FX_DAILY"
    request_endpoint = f"https://www.alphavantage.co/query?function={function}&from_symbol={from_symbol}&to_symbol={to_symbol}&apikey={apikey}&outputsize={outputsize}&datatype={datatype}"
    
    daily_fx_data_df = fetch_data(request_endpoint, function, datatype)

    df_subset = daily_fx_data_df[[dict_price_types[price_type]]]
    df_subset = df_subset.rename(columns={dict_price_types[price_type]: f'{price_type.upper()}'})

    return df_subset

##### COMMODITIES APIs #######

def extract_commodity_data(commodity_function, interval='monthly', datatype='json', apikey="DEESUXPJ76GP18UB"):
    """
    Fetches commodity data from the Alpha Vantage API.

    Parameters:
    - apikey: Your Alpha Vantage API key.
    - commodity_function: The specific commodity data function ('WTI', 'BRENT', 'NATURAL_GAS', 'COPPER', 'ALUMINUM', 'WHEAT', 'CORN', 'COTTON', 'SUGAR', 'COFFEE', 'ALL_COMMODITIES').
    - interval: The time interval for the data ('daily', 'weekly', 'monthly', 'quarterly', 'annual').
    - datatype: Format of the returned data ('json' or 'csv').

    Returns:
    - A pandas DataFrame containing the requested commodity data.
    """

    request_endpoint = f"https://www.alphavantage.co/query?function={commodity_function}&interval={interval}&apikey={apikey}&datatype={datatype}"

    commodity_data_df = fetch_data(request_endpoint, commodity_function, datatype)

    return commodity_data_df
    


##### Filling APIs #######

def extract_filling_data(filling_api_identity, CompanyTicker, StartDate, EndDate, FillingType, FillingSection, extract_numeric):

    set_identity(filling_api_identity)

    filing_obj = Company(CompanyTicker).get_filings(form=FillingType).filter(date=f"{StartDate}:{EndDate}")[0].obj()
    if FillingSection in filing_obj.items:
        filing_text = Company(CompanyTicker).get_filings(form=FillingType).filter(date=f"{StartDate}:{EndDate}")[0].text()
    else:
        filing_text = f"No corresponding {FillingSection} in {FillingType} for {CompanyTicker} between {StartDate} and {EndDate}"

    if FillingType == "10-K":
        balance_sheet = filing_obj.financials.balance_sheet.to_dataframe()
        cash_flow_statement = filing_obj.financials.cash_flow_statement.to_dataframe()
        income_statement = filing_obj.financials.income_statement.to_dataframe()
    
    
    if extract_numeric:
        return filing_text, [balance_sheet, cash_flow_statement, income_statement]
    else:
        return filing_text, None


def get_filing_text(filing_api_identity, ticker, year, filing_type, section):
    start_date = datetime.date(int(year), 1, 1)
    end_date = datetime.date(int(year), 12, 31)
    
    # Assuming extract_filling_data is a function that fetches the filing data correctly
    filing_text, _ = extract_filling_data(filing_api_identity, ticker, start_date, end_date, filing_type, section, extract_numeric=False)
    
    # Define a file path where the text will be saved
    file_path = f"/home/yasser/SM_project/filings/{ticker}_{year}_{filing_type}_{section}.txt"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Write the text to a file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(filing_text)
    
    # Return the path to the saved file
    if filing_text:
        return file_path, True
    else:
        return file_path, False




