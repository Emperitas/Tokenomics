import wbdata
from fredapi import Fred
import time
from typing import List, Dict
import pandas as pd
import numpy as np
from sqlalchemy import Engine
import logging
from pmdarima import auto_arima
import yfinance as yf


# Constants
DATE_COLUMN = "date"
DATE_NEW_COLUMN = "date_new"
logging.basicConfig(
    level=logging.DEBUG,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)


def get_yahoo_data(indicators: List, freq:str) -> List[pd.DataFrame]:
    """
    Fetch data from Yahoo Finance API based on provided configuration.

    Args:
        indicators: List of ticker symbols to fetch
        freq: Frequency of data to fetch (e.g., '1d', '1wk', '1mo')

    Returns:
        List of pandas DataFrames containing the fetched data
    """
    dfs = []
    for ticker in indicators:
        try:
            logging.info(f"Downloading {ticker} from Yahoo Finance")
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(period="max", interval=freq)

            if df.empty:
                logging.warning(f"No data found for {ticker}")
                continue

            # Ensure all required columns are present
            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            if not all(col in df.columns for col in required_columns):
                logging.warning(f"Missing required columns for {ticker}. Available columns: {df.columns.tolist()}")
                # Use available columns or fill missing ones with NaN
                for col in required_columns:
                    if col not in df.columns:
                        df[col] = np.nan

            # Select only the required columns
            ticker_dfs = []
            for c in required_columns:
                data = df[c].to_frame()
                data.attrs["table_name"] = f"yahoo_{ticker}_{c}"
                ticker_dfs.append(data)

            dfs.extend(ticker_dfs)
            logging.info(f"Successfully downloaded {ticker} data with shape {df.shape}")
        except Exception as e:
            logging.error(f"Error downloading {ticker} from Yahoo Finance: {str(e)}")

    return dfs


def fetch_world_bank_data(indicators: Dict, country: str) -> List[pd.DataFrame]:
    """
    Fetch data from World Bank API based on provided configuration.

    Args:
        indicators: Dictionary mapping World Bank indicator codes to names
        country: Country code to fetch data for

    Returns:
        List of pandas DataFrames containing the fetched data
    """
    dfs = []
    try:
        logging.info(f"Fetching World Bank data for country {country} with indicators: {indicators}")

        # Attempt to fetch data from World Bank API
        df = wbdata.get_dataframe(
            indicators,
            parse_dates=True,
            freq="M",
            date=("2000-01-01", "2025-12-31"),
            skip_cache=True,
            country=country,
        )

        if df.empty:
            logging.warning(f"No World Bank data found for country {country}")
            return dfs

        # Process each indicator separately
        for col in df.columns:
            try:
                data = df[col].to_frame()

                # Skip if all values are NaN
                if data[col].isna().all():
                    logging.warning(f"No data for indicator {col} in country {country}")
                    continue

                # Set table name attribute
                data.attrs["table_name"] = f"world_bank_{country}_{col}"

                dfs.append(data)
                logging.info(f"Successfully processed World Bank data for {col} with shape {data.shape}")
            except Exception as e:
                logging.error(f"Error processing World Bank data for indicator {col}: {str(e)}")

    except Exception as e:
        logging.error(f"Error fetching World Bank data for country {country}: {str(e)}")

    return dfs


def get_fred_data(fred_instance: Fred, series_name: Dict) -> List[pd.DataFrame]:
    """
    Downloads data from FREDAPI with selected series.

    Args:
        fred_instance: Instance of the FRED API client
        series_name: Dictionary mapping series names to FRED series IDs

    Returns:
        List of pandas DataFrames containing the fetched data
    """
    dfs = []
    for name, series_id in series_name.items():
        try:
            logging.info(f"Downloading {name} (series ID: {series_id}) from FRED")

            # Add a delay to avoid hitting API rate limits
            time.sleep(2)

            # Fetch the series data
            series = fred_instance.get_series(
                series_id, observation_start="1999-12-31"
            )

            # Check if series is empty
            if series.empty:
                logging.warning(f"No data found for FRED series {name} (ID: {series_id})")
                continue

            # Convert to DataFrame
            df = series.to_frame()
            df.columns = [name]
            df["date"] = df.index

            # Set table name attribute
            df.attrs["table_name"] = f"fred_{name}"

            dfs.append(df)
            logging.info(f"Successfully downloaded FRED data for {name} with shape {df.shape}")

        except Exception as e:
            logging.error(f"Error downloading FRED series {name} (ID: {series_id}): {str(e)}")

    return dfs


def store_data_in_db(engine: Engine, dfs: List[pd.DataFrame]) -> None:
    """Stores data in a CSV file"""
    for item in dfs:
        item.to_sql(item.attrs["table_name"], engine, if_exists="replace", index=True)


def reindex_data(
    dataframes: List[pd.DataFrame], start_date: str, end_date: str, frequency: str
) -> List[pd.DataFrame]:
    """
    Reindex multiple dataframes with specified date range and frequency.

    Args:
        dataframes: List of pandas DataFrames to reindex
        start_date: Start date for the new date range
        end_date: End date for the new date range
        frequency: Frequency for the new date range (e.g., 'D', 'W', 'M', 'Y')

    Returns:
        List of reindexed pandas DataFrames
    """
    return [
        reindex_single_dataframe(df, start_date, end_date, frequency)
        for df in dataframes
    ]


def reindex_single_dataframe(
    df: pd.DataFrame, start_date: str, end_date: str, frequency: str
) -> pd.DataFrame:
    """
    Reindex a single dataframe with specified date range and frequency.
    Handles different data sources appropriately.

    Args:
        df: pandas DataFrame to reindex
        start_date: Start date for the new date range
        end_date: End date for the new date range
        frequency: Frequency for the new date range (e.g., 'D', 'W', 'M', 'Y')

    Returns:
        Reindexed pandas DataFrame
    """
    logging.debug(
        f"Reindexing {df.attrs.get('table_name', 'unknown')} to {start_date} - {end_date} with frequency {frequency}"
    )

    # Create a copy to avoid modifying the original
    df_copy = df.copy()

    # Create target date range
    target_dates = pd.date_range(start_date, end_date, freq=frequency)

    # Handle different data sources
    data_source = ""
    if hasattr(df, 'attrs') and 'table_name' in df.attrs:
        table_name = df.attrs['table_name']
        if "world_bank" in table_name:
            data_source = "world_bank"
        elif "yahoo" in table_name or "10y_treasury" in table_name or "nasdaq_com" in table_name:
            data_source = "yahoo"
        elif "fred" in table_name:
            data_source = "fred"


    # Ensure df has a DatetimeIndex
    if 'date' in df_copy.columns:
        df_copy = df_copy.set_index('date')

    if not isinstance(df_copy.index, pd.DatetimeIndex):
        try:
            df_copy.index = pd.to_datetime(df_copy.index)
        except:
            logging.warning(f"Could not convert index to DatetimeIndex for {df.attrs.get('table_name', 'unknown')}")
            return df_copy

    # Handle Yahoo Finance and high-frequency FRED data
    if data_source == "yahoo" or (data_source == "fred" and ("10y_treasury" in df.attrs.get('table_name', '') or "nasdaq_com" in df.attrs.get('table_name', ''))):
        # For Yahoo Finance data, resample to weekly frequency
        if "Open" in df.columns:
            resampled = df_copy.resample(frequency).agg({"Open": "first"})
        elif "High" in df.columns:
            resampled = df_copy.resample(frequency).agg({"High": "max"})
        elif "Low" in df.columns:
            resampled = df_copy.resample(frequency).agg({"Low": "min"})
        elif "Close" in df.columns:
            resampled = df_copy.resample(frequency).agg({"Close": "last"})
        elif "Volume" in df.columns:
            resampled = df_copy.resample(frequency).agg({"Volume": "sum"})
        else:
            # For other data, use mean
            resampled = df_copy.resample(frequency).mean()

        # Preserve the table_name attribute
        if hasattr(df, 'attrs') and 'table_name' in df.attrs:
            resampled.attrs['table_name'] = df.attrs['table_name']

        return resampled

    # For World Bank and FRED data, reindex to target dates and interpolate
    # Reindex to target dates
    reindexed = df_copy.reindex(target_dates, )

    # Handle country information for World Bank data
    if data_source == "world_bank":
        # Add country information if available
        if 'country' in df_copy.columns:
            reindexed['country'] = df_copy['country'].iloc[0] if not df_copy.empty else None
        elif hasattr(df, 'attrs') and 'table_name' in df.attrs:
            table_name = df.attrs['table_name']
            if 'world_bank_' in table_name:
                parts = table_name.split('_')
                if len(parts) > 2:
                    country_code = parts[2]
                    reindexed['country'] = country_code

    # Interpolate missing values
    if "federal_rate" in df.attrs["table_name"]:
        reindexed = reindexed.ffill()
    else:
        reindexed = reindexed.interpolate(method='time')

    # Preserve the table_name attribute
    if hasattr(df, 'attrs') and 'table_name' in df.attrs:
        reindexed.attrs['table_name'] = df.attrs['table_name']

    return reindexed


def find_best_arima(data: pd.DataFrame):
    """
    Find the best ARIMA model for the given data.

    Args:
        data: A pandas DataFrame or Series containing the time series data

    Returns:
        A fitted ARIMA model
    """
    # If data is a DataFrame with multiple columns, select a single column
    if isinstance(data, pd.DataFrame):
        if data.shape[1] > 1:
            raise ValueError("Data must be a DataFrame with a single column for ARIMA modeling")
        else:
            # If data is a DataFrame with a single column, extract the series
            column_name = data.columns[0]
            logging.info(f"Using '{column_name}' column for ARIMA modeling")
            data = data[column_name]

    # Fit auto_arima with more reasonable parameter ranges
    # Reduced max_p, max_q, and max_d to prevent overfitting and improve performance
    model = auto_arima(
        data,
        max_p=10,
        max_q=10,
        max_d=10,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        max_order=None,
        n_jobs=-1,
        #method="bfgs",
        information_criterion="aicc",

    )

    logging.info(f"  Best order: {model.order} (AIC: {model.aic():.2f})")
    return model


def analyze_with_auto_arima(df, value_column="value"):
    """
    Analyze time series data with auto_arima to find the best ARIMA model.

    Args:
        df: DataFrame containing time series data
        value_column: Column name containing the values to analyze

    Returns:
        Dictionary of ARIMA model results
    """
    results = {}

    # Skip empty dataframes
    df = df.dropna()
    if df.empty:
        logging.warning(f"Empty DataFrame {df.attrs.get('table_name', 'unknown')} after dropping NaN values")
        return results

    # Get table name for logging
    table_name = df.attrs.get('table_name', 'unknown')

    # Check if the specified value_column exists in the DataFrame
    if value_column not in df.columns:
        # If not, use the first available column that is not 'date' or 'country'
        available_columns = [col for col in df.columns if col not in [DATE_COLUMN, 'country']]
        if available_columns:
            value_column = available_columns[0]
        else:
            logging.error(f"No suitable value column found in DataFrame {table_name}")
            return results

    # Ensure DataFrame has a DatetimeIndex
    if DATE_COLUMN in df.columns:
        df = df.set_index(DATE_COLUMN)
    elif not isinstance(df.index, pd.DatetimeIndex):
        logging.warning(f"DataFrame {table_name} does not have a date column or DatetimeIndex")
        return results

    # Make sure the data is sorted by date
    df = df.sort_index()

    # Handle different data sources
    if "fred" in table_name:
        # For FRED data, resample to yearly data first if specified
        model = find_best_arima(df)

        results[table_name] = {
            "order": model.order,
            "aic": model.aic(),
            "model": model,
        }
    elif "yahoo" in table_name:
        # For Yahoo Finance data, create a model for each column
        column_df = df.copy()
        model = find_best_arima(column_df)
        results[f"{table_name}"] = {
            "order": model.order,
            "aic": model.aic(),
            "model": model,
            "is_yearly": False
        }
    elif "world_bank" in df.attrs["table_name"] and "country" in df.columns:
        # Get unique countries
        countries = list(df["country"].unique())
        for country in countries:
            try:
                # Extract data for this country
                country_df = df[df.country == country]

                # Extract the data series
                if value_column in country_df.columns:
                    country_data = country_df[value_column]
                else:
                    # Use the first column if value_column is not available
                    country_data = country_df.iloc[:, 0]

                # Make sure the data is sorted by index
                country_data = country_data.sort_index()
                model = find_best_arima(country_data)

                # Store results with a more descriptive key
                # Use the table_name as a prefix to avoid creating a column named after the country
                indicator_name = table_name.split('_')[-1] if '_' in table_name else 'unknown'
                results[f"world_bank_{country}_{indicator_name}"] = {
                    "order": model.order,
                    "aic": model.aic(),
                    "model": model,
                }
            except Exception as e:
                logging.error(f"Error analyzing data for country {country}: {str(e)}")
    else:
        # Handle single series (no country information)
        try:
            # Extract the data series
            if isinstance(df, pd.DataFrame):
                if value_column in df.columns:
                    data = df[value_column]
                else:
                    # Use the first column if value_column is not available
                    data = df.iloc[:, 0]
            else:
                data = df  # Already a Series

            # Make sure the data is sorted by date
            data = data.sort_index()

            model = find_best_arima(data)

            # Store results
            results[table_name] = {
                "order": model.order,
                "aic": model.aic(),
                "model": model,
            }
        except Exception as e:
            logging.error(f"Error analyzing data for {table_name}: {str(e)}")

    return results


def make_predictions(results: dict, steps: int = 10):
    """
    Make predictions using ARIMA models, with special handling for yearly vs. weekly data.

    Args:
        results: Dictionary of ARIMA model results
        steps: Number of steps (weeks) to predict

    Returns:
        Dictionary of forecasts
    """
    forecasts = []

    for table, result in results.items():
        model = result["model"]
        step_length = pd.DataFrame(model.predict(steps=2))
        diff = (step_length.index[2] - step_length.index[1])
        if isinstance(diff, pd.Timedelta):
            diff = diff.days
        logging.debug(f"Step length for {table}: {diff}")
        if diff < 7:
            forecast_values = model.predict(n_periods=steps * 365)
        elif 6 < diff < 8:
            forecast_values = model.predict(n_periods=steps * 52)
        elif 25 < diff < 35:
            forecast_values = model.predict(n_periods=steps * 12)
        elif 35 < diff < 100:
            forecast_values = model.predict(n_periods=steps * 4)
        else:
            forecast_values = model.predict(n_periods=steps)
        forecast_values.attrs["table_name"] = f"{table}_forecast"
        forecasts.append(forecast_values)
        logging.debug(f"Sample forecast for {table}: {forecast_values.head(5)}")

    return forecasts


def combine_dataframes(dataframes: List[pd.DataFrame], forecasts: List) -> pd.DataFrame:
    """
    Combines multiple dataframes into one final table, optionally including forecasts.
    Excludes redundant columns and ensures all data is resampled to weekly frequency.

    Args:
        dataframes: List of pandas DataFrames to combine
        forecasts: Dictionary of forecast Series

    Returns:
        A single pandas DataFrame containing all the data with dates as indexes
    """
    if not dataframes:
        return pd.DataFrame()

    # Step 1: Prepare dataframes for combination
    processed_dfs = []
    for df in dataframes:
        # Skip if empty
        if df.empty:
            continue

        # Make a copy to avoid modifying the original
        df_copy = df.copy()

        # Set date as index if it's a column
        if 'date' in df_copy.columns:
            df_copy = df_copy.set_index('date')

        # Skip if not a DatetimeIndex
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            logging.warning(f"DataFrame {df_copy.attrs.get('table_name', 'unknown')} does not have a date index or column")
            continue

        # Convert timezone-aware timestamps to timezone-naive
        if df_copy.index.tz is not None:
            df_copy.index = df_copy.index.tz_localize(None)

        # Remove duplicate indices
        if df_copy.index.duplicated().any():
            logging.warning(f"DataFrame {df_copy.attrs.get('table_name', 'unknown')} has duplicate indices. Taking the last value for each date.")
            df_copy = df_copy[~df_copy.index.duplicated(keep='last')]

        # Preserve the table_name attribute
        if hasattr(df, 'attrs') and 'table_name' in df.attrs:
            df_copy.attrs['table_name'] = df.attrs['table_name']

        processed_dfs.append(df_copy)

    # Step 2: Collect all dates and create a common date range
    all_dates = set()

    # Add dates from dataframes
    for df in processed_dfs:
        all_dates.update(df.index)

    # Add dates from forecasts
    if forecasts:
        for forecast in forecasts:
            if isinstance(forecast.index, pd.DatetimeIndex):
                # Convert timezone-aware timestamps to timezone-naive
                if forecast.index.tz is not None:
                    forecast_index = forecast.index.tz_localize(None)
                else:
                    forecast_index = forecast.index

                all_dates.update(forecast_index)

    # Create an empty dataframe with the common date range
    combined_df = pd.DataFrame(index=sorted(all_dates))
    combined_df.index.name = 'date'

    # Step 3: Add data columns from each dataframe
    added_columns = set()

    # Add country information if available
    for df in processed_dfs:
        table_name = df.attrs.get('table_name', 'unknown')
        if 'world_bank_' in table_name and 'country_code' not in added_columns:
            if 'country' in df.columns:
                # Rename to country_code for clarity
                combined_df['country_code'] = df['country']
                added_columns.add('country_code')
            else:
                # Extract country code from table_name
                country_parts = table_name.replace("world_bank_","").split('_')
                if len(country_parts) > 2:
                    country_code = country_parts[2]
                    combined_df['country_code'] = country_code
                    added_columns.add('country_code')

    # Add data columns
    for df in processed_dfs:
        table_name = df.attrs.get('table_name', 'unknown')

        for column in df.columns:
            # Skip redundant columns
            if column.lower() in ['date', 'date_new', f'old_{DATE_COLUMN}', 'country', 'country_code']:
                continue

            # Create a descriptive column name
            if 'world_bank_' in table_name:
                new_column_name = table_name.replace("world_bank_", "")  # Get the indicator name
            elif 'fred_' in table_name:
                new_column_name = table_name.replace('fred_', '')
            elif 'yahoo_' in table_name:
                ticker = table_name.replace('yahoo_', '')
                new_column_name = f"{ticker}"
            else:
                new_column_name = f"{table_name}_{column}"

            # Skip if we already have this column
            if new_column_name in added_columns:
                continue

            # Add the column to the combined dataframe
            combined_df[new_column_name] = df[column]
            added_columns.add(new_column_name)

    # Step 4: Add forecast data if available
    if forecasts:
        # Get the current date to separate historical data from forecast data
        current_date = pd.Timestamp.now()

        for forecast in forecasts:
            # Identify the corresponding base column for this forecast
            base_column = None
            table_name = forecast.attrs.get('table_name', 'unknown')
            if 'world_bank_' in table_name:
                base_column = table_name.replace("world_bank_", "")  # Get the indicator name
            elif 'fred_' in table_name:
                base_column = table_name.replace('fred_', '')
            elif 'yahoo_' in table_name:
                # Handle Yahoo Finance data
                if '_' in table_name.replace('yahoo_', ''):
                    # Case: yahoo_BTC-USD_Open, yahoo_ETH-USD_High, etc.
                    parts = table_name.replace('yahoo_', '').split('_')
                    ticker = parts[0]
                    column = parts[1] if len(parts) > 1 else ''
                    base_column = f"{ticker}_{column}"
            else:
                # For other cases, try to find a matching column without "_forecast" suffix
                possible_base = table_name
                if possible_base in combined_df.columns:
                    base_column = possible_base

            # If we found a corresponding base column, merge the forecast with it
            if base_column and base_column in combined_df.columns:
                # Create a mask for future dates (after current date)
                future_mask = combined_df.index > current_date

                # Get the forecast values for future dates
                # Handle both Series and DataFrame forecasts
                if isinstance(forecast, pd.DataFrame):
                    # If forecast is a DataFrame, get the first column
                    forecast_column = forecast.columns[0]
                    future_forecast = forecast.loc[forecast.index.isin(combined_df.index[future_mask]), forecast_column]
                else:
                    # If forecast is a Series, use it directly
                    future_forecast = forecast[forecast.index.isin(combined_df.index[future_mask])]

                # Assign the forecast values to the base column for future dates
                combined_df.loc[future_mask, base_column] = future_forecast

                logging.info(f"Merged forecast for {table_name} with base column {base_column}")
            else:
                # If no corresponding base column found, add as a new column
                logging.warning(f"No matching base column found for {table_name}, adding as a new column")

                # Create a descriptive column name for the forecast
                if 'world_bank_' in table_name:
                    new_column = table_name.replace('world_bank_', "")
                elif 'fred_' in table_name:
                    new_column = table_name.replace('fred_', '')
                elif 'yahoo_' in table_name and '_' in table_name:
                    parts = table_name.replace('yahoo_', '').split('_')
                    ticker = parts[0]
                    column = parts[1] if len(parts) > 1 else ''
                    new_column = f"{ticker}_{column}".replace("-","_")
                else:
                    new_column = table_name

                # Add the forecast to the combined dataframe
                # Handle both Series and DataFrame forecasts
                if isinstance(forecast, pd.DataFrame):
                    # If forecast is a DataFrame, get the first column
                    forecast_column = forecast.columns[0]
                    combined_df[new_column] = forecast[forecast_column]
                else:
                    # If forecast is a Series, use it directly
                    combined_df[new_column] = forecast

                added_columns.add(new_column)
                logging.info(f"Added forecast as new column: {new_column}")

    # Step 7: Sort the combined dataframe by date
    combined_df = combined_df.sort_index()

    # Set the table name attribute
    combined_df.attrs['table_name'] = 'combined_data'

    return combined_df
