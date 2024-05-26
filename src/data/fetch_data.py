import pandas as pd
from tqdm import tqdm

from utils import *
# TIME SERIES DATA

symbols = ["IBM", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK.B", "GOOG", "LLY", "AVGO", "JPM", "TSLA", "V", "UNH", "XOM", "MA", "JNJ", "HD", "PG", "COST"]

# symbols = ["IBM", "AAPL"]

daily_times_series_data_processed = pd.DataFrame()

for symbol in tqdm(symbols):

    # Technical Indicators

    daily_technical_indicators_df = pd.DataFrame()

    for technical_indicator in ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'MAMA', 'T3', 'MACD', 'MACDEXT', 'STOCH', 'STOCHF', 'RSI', 'STOCHRSI', 'WILLR', 'ADX', 'ADXR', 'APO', 'PPO', 'MOM', 'BOP', 'CCI', 'CMO', 'ROC', 'ROCR', 'AROON', 'AROONOSC', 'MFI', 'TRIX', 'ULTOSC', 'DX', 'MINUS_DI', 'PLUS_DI', 'MINUS_DM', 'PLUS_DM', 'BBANDS', 'MIDPOINT', 'MIDPRICE', 'SAR', 'TRANGE', 'ATR', 'NATR', 'AD', 'ADOSC', 'OBV', 'HT_TRENDLINE', 'HT_SINE', 'HT_TRENDMODE', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR']:
        try:
            # Call your API function to get the DataFrame for the current technical indicator
            technical_indicator_df = extract_technical_indicators_api(function=technical_indicator, symbol=symbol, interval="daily", time_period=60, series_type="close", apikey="")
            
            # Check if master_df is empty (first iteration)
            if daily_technical_indicators_df.empty:
                daily_technical_indicators_df = technical_indicator_df
            else:
                # Merge the new DataFrame into the master DataFrame based on index
                daily_technical_indicators_df = pd.merge(daily_technical_indicators_df, technical_indicator_df, left_index=True, right_index=True, how='outer')
        except Exception as e:
            print(f"Failed to process {technical_indicator}: {e}")

    # Daily Storck prices

    daily_data_df = extract_daily_data(symbol=symbol, adjusted=True, outputsize="full", datatype="json", apikey="")
    daily_data_df = daily_data_df.rename(columns={'1. open': 'OPEN_PRICE', '2. high': 'HIGH_PRICE', '3. low': 'LOW_PRICE', '4. close': 'CLOSE_PRICE', '5. adjusted close': 'ADJUSTED_CLOSE_PRICE', '6. volume': 'VOLUME', '7. dividend amount': 'DIVIDEND_AMOUNT', '8. split coefficient': 'SPLIT_COEFFICIENT'})

    # Daily Storck prices

    to_symbol = "USD"
    daily_forex_data_df = pd.DataFrame()

    for from_symbol in ['EUR', 'GBP', 'CNH', 'CNY', 'JPY']:
        try:
            # Call your API function to get the DataFrame for the current Forex Value
            daily_forex_data_individual_df = extract_fx_daily_data(from_symbol, to_symbol, outputsize="full", datatype="json", apikey="")
            
            # Rename the '4. close' column to reflect the currency pair, e.g., "EUR_USD"
            daily_forex_data_individual_df = daily_forex_data_individual_df.rename(columns={'4. close': f'{from_symbol}_{to_symbol}'})
            
            # Check if master_df is empty (first iteration)
            if daily_forex_data_df.empty:
                daily_forex_data_df = daily_forex_data_individual_df[f'{from_symbol}_{to_symbol}']
            else:
                # Merge the new DataFrame into the master DataFrame based on index
                # Since we're merging the entire DataFrame now, no need to specify '4. close'
                daily_forex_data_df = pd.merge(daily_forex_data_df, daily_forex_data_individual_df[f'{from_symbol}_{to_symbol}'], left_index=True, right_index=True, how='outer')
        except Exception as e:
            print(f"Failed to process From {from_symbol} to USD: {e}")

    # Economic Indicators in the US

    daily_economic_indicators_data_df = pd.DataFrame()

    for economic_indicator_function in ["TREASURY_YIELD", "FEDERAL_FUNDS_RATE"]:
        try:
            # Call your API function to get the DataFrame for the current Forex Value
            economic_indicator_data_df = extract_economic_indicators_api(function=economic_indicator_function, datatype="json", apikey="")
            
            # Rename the '4. close' column to reflect the currency pair, e.g., "EUR_USD"
            economic_indicator_data_df = economic_indicator_data_df.rename(columns={'value': f'{economic_indicator_function}'})
            
            # Check if master_df is empty (first iteration)
            if economic_indicator_data_df.empty:
                daily_economic_indicators_data_df = economic_indicator_data_df[f'{economic_indicator_function}']
            else:
                # Merge the new DataFrame into the master DataFrame based on index
                daily_economic_indicators_data_df  = pd.merge(daily_economic_indicators_data_df, economic_indicator_data_df[f'{economic_indicator_function}'], left_index=True, right_index=True, how='outer')
        except Exception as e:
            print(f"Failed to process {economic_indicator_function}: {e}")

    monthly_economic_indicators_data_df = pd.DataFrame()

    for economic_indicator_function in ["CPI", "RETAIL_SALES", "DURABLES", "UNEMPLOYMENT", "NONFARM_PAYROLL"]:
        try:
            # Call your API function to get the DataFrame for the current Forex Value
            economic_indicator_data_df = extract_economic_indicators_api(function=economic_indicator_function, datatype="json", apikey="")
            
            # Rename the '4. close' column to reflect the currency pair, e.g., "EUR_USD"
            economic_indicator_data_df = economic_indicator_data_df.rename(columns={'value': f'{economic_indicator_function}'})
            
            # Check if master_df is empty (first iteration)
            if economic_indicator_data_df.empty:
                monthly_economic_indicators_data_df = economic_indicator_data_df[f'{economic_indicator_function}']
            else:
                # Merge the new DataFrame into the master DataFrame based on index
                monthly_economic_indicators_data_df  = pd.merge(monthly_economic_indicators_data_df, economic_indicator_data_df[f'{economic_indicator_function}'], left_index=True, right_index=True, how='outer')
        except Exception as e:
            print(f"Failed to process {economic_indicator_function}: {e}")

    quarterly_economic_indicators_data_df = pd.DataFrame()

    for economic_indicator_function in ["REAL_GDP", "REAL_GDP_PER_CAPITA"]:
        try:
            # Call your API function to get the DataFrame for the current Forex Value
            economic_indicator_data_df = extract_economic_indicators_api(function=economic_indicator_function, datatype="json", apikey="")
            
            # Rename the '4. close' column to reflect the currency pair, e.g., "EUR_USD"
            economic_indicator_data_df = economic_indicator_data_df.rename(columns={'value': f'{economic_indicator_function}'})
            
            # Check if master_df is empty (first iteration)
            if economic_indicator_data_df.empty:
                quarterly_economic_indicators_data_df = economic_indicator_data_df[f'{economic_indicator_function}']
            else:
                # Merge the new DataFrame into the master DataFrame based on index
                quarterly_economic_indicators_data_df  = pd.merge(quarterly_economic_indicators_data_df, economic_indicator_data_df[f'{economic_indicator_function}'], left_index=True, right_index=True, how='outer')
        except Exception as e:
            print(f"Failed to process {economic_indicator_function}: {e}")

    annual_economic_indicators_data_df = pd.DataFrame()

    for economic_indicator_function in ["INFLATION"]:
        try:
            # Call your API function to get the DataFrame for the current Forex Value
            economic_indicator_data_df = extract_economic_indicators_api(function=economic_indicator_function, datatype="json", apikey="")
            
            # Rename the '4. close' column to reflect the currency pair, e.g., "EUR_USD"
            economic_indicator_data_df = economic_indicator_data_df.rename(columns={'value': f'{economic_indicator_function}'})
            
            # Check if master_df is empty (first iteration)
            if economic_indicator_data_df.empty:
                annual_economic_indicators_data_df = economic_indicator_data_df[f'{economic_indicator_function}']
            else:
                # Merge the new DataFrame into the master DataFrame based on index
                annual_economic_indicators_data_df  = pd.merge(annual_economic_indicators_data_df, economic_indicator_data_df[f'{economic_indicator_function}'], left_index=True, right_index=True, how='outer')
        except Exception as e:
            print(f"Failed to process {economic_indicator_function}: {e}")



    # Commodities

    daily_commodities_data_df = pd.DataFrame()

    for commodity_function in ['WTI', 'BRENT', 'NATURAL_GAS']:
        try:
            # Call your API function to get the DataFrame for the current Forex Value
            commodity_data_df = extract_commodity_data(commodity_function, interval='daily', datatype='json', apikey="")
            
            # Rename the '4. close' column to reflect the currency pair, e.g., "EUR_USD"
            commodity_data_df = commodity_data_df.rename(columns={'value': f'{commodity_function}'})
            
            # Check if master_df is empty (first iteration)
            if daily_commodities_data_df.empty:
                daily_commodities_data_df = commodity_data_df[f'{commodity_function}']
            else:
                # Merge the new DataFrame into the master DataFrame based on index
                daily_commodities_data_df  = pd.merge(daily_commodities_data_df , commodity_data_df[f'{commodity_function}'], left_index=True, right_index=True, how='outer')
        except Exception as e:
            print(f"Failed to process {commodity_function}: {e}")


    monthly_commodities_data_df = pd.DataFrame()

    for commodity_function in ['COPPER', 'ALUMINUM', 'WHEAT', 'CORN', 'COTTON', 'SUGAR', 'COFFEE', 'ALL_COMMODITIES']:
        try:
            # Call your API function to get the DataFrame for the current Forex Value
            commodity_data_df = extract_commodity_data(commodity_function, interval='monthly', datatype='json', apikey="")
            
            # Rename the '4. close' column to reflect the currency pair, e.g., "EUR_USD"
            commodity_data_df = commodity_data_df.rename(columns={'value': f'{commodity_function}'})
            
            # Check if master_df is empty (first iteration)
            if monthly_commodities_data_df.empty:
                monthly_commodities_data_df = commodity_data_df[f'{commodity_function}']
            else:
                # Merge the new DataFrame into the master DataFrame based on index
                # Since we're merging the entire DataFrame now, no need to specify '4. close'
                monthly_commodities_data_df  = pd.merge(monthly_commodities_data_df , commodity_data_df[f'{commodity_function}'], left_index=True, right_index=True, how='outer')
        except Exception as e:
            print(f"Failed to process {commodity_function}: {e}")





    commodities_data_df = pd.merge(monthly_commodities_data_df.resample('D').ffill(), daily_commodities_data_df, left_index=True, right_index=True, how='outer')
    economic_indicators_data_df = pd.merge(pd.merge(pd.merge(monthly_economic_indicators_data_df.resample('D').ffill(), daily_economic_indicators_data_df, left_index=True, right_index=True, how='outer'),quarterly_economic_indicators_data_df.resample('D').ffill(),left_index=True, right_index=True, how='outer'), annual_economic_indicators_data_df.resample('D').ffill(),left_index=True, right_index=True, how='outer')

    daily_times_series_data = merge_data_frames(daily_data_df, 
                            (daily_forex_data_df, "D"),
                            (commodities_data_df, "D"),
                            (daily_technical_indicators_df, "D"),
                            (economic_indicators_data_df, "D"))


    daily_times_series_data_processed_per_symbol =  daily_times_series_data[daily_times_series_data.index > '1999-11-01'].interpolate(method='time')

    daily_times_series_data_processed_per_symbol['SYMBOL'] = symbol

    daily_times_series_data_processed_per_symbol['DAY_PLUS_1_ADJUSTED_CLOSING_PRICE'] = daily_times_series_data_processed_per_symbol['ADJUSTED_CLOSE_PRICE'].shift(1)
    daily_times_series_data_processed_per_symbol['DAY_PLUS_2_ADJUSTED_CLOSING_PRICE'] = daily_times_series_data_processed_per_symbol['ADJUSTED_CLOSE_PRICE'].shift(2)
    daily_times_series_data_processed_per_symbol['DAY_PLUS_3_ADJUSTED_CLOSING_PRICE'] = daily_times_series_data_processed_per_symbol['ADJUSTED_CLOSE_PRICE'].shift(3)
    daily_times_series_data_processed_per_symbol['DAY_PLUS_4_ADJUSTED_CLOSING_PRICE'] = daily_times_series_data_processed_per_symbol['ADJUSTED_CLOSE_PRICE'].shift(4)
    daily_times_series_data_processed_per_symbol['DAY_PLUS_5_ADJUSTED_CLOSING_PRICE'] = daily_times_series_data_processed_per_symbol['ADJUSTED_CLOSE_PRICE'].shift(5)

    daily_times_series_data_processed_per_symbol = daily_times_series_data_processed_per_symbol.iloc[6:]

    # Fill missing values with forward fill method, then backfill if needed
    daily_times_series_data_processed_per_symbol.fillna(method='ffill', inplace=True)
    daily_times_series_data_processed_per_symbol.fillna(method='bfill', inplace=True)

    if daily_times_series_data_processed.empty:
        daily_times_series_data_processed = daily_times_series_data_processed_per_symbol
    else:
        daily_times_series_data_processed = pd.concat([daily_times_series_data_processed, daily_times_series_data_processed_per_symbol])

# Turn the index into a column
daily_times_series_data_processed_final = daily_times_series_data_processed.reset_index()

# Optionally, you can rename the 'index' column to something more meaningful
daily_times_series_data_processed_final.rename(columns={'index': 'DATE'}, inplace=True)


daily_times_series_data_processed.to_csv("/home/yasser/SM_project/src/data/time_series_data.csv", index=True) 

