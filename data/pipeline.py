from etl import get_fred_data, fetch_world_bank_data, reindex_data, store_data_in_db, analyze_with_auto_arima, make_predictions, get_yahoo_data, combine_dataframes
import os
from fredapi import Fred
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging
import pandas as pd

logging.basicConfig(
    level=logging.DEBUG,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)

load_dotenv()

fred_indicators = {
    "cpi": "CPIAUCSL",
    "federal_rate": "FEDFUNDS",
    "m2": "M2SL",
    "10y_treasury": "DGS10",
    "nasdaq_com": "NASDAQCOM",
    "corp_profits": "CP",
    "telecom_investments": "Y033RC1Q027SBEA",
}

yahoo_indicators = ["BTC-USD", "ETH-USD"]

wb_indicators = {
    "NY.GDP.PCAP.CD": "gdp_growth",
    "BX.KLT.DINV.CD.WD": "fdi_net_inflows",
    "IC.BUS.DFRN.XQ": "business_ease",
    "IT.CEL.SETS.P2": "cell_subs",
    "IT.NET.BBND.P2": "broadband_subs",
    "TX.VAL.ICTG.ZS.UN": "ict_goods_imports",
    "IP.PAT.RESD": "patent_apps",
    "GB.XPD.RSDV.GD.ZS": "rd_expenditure",
}


def get_data(from_db:bool = False):
    """
    Main function to extract, transform, and load data from various sources.

    Args:
        from_db: If True, load data from database instead of fetching from APIs

    Returns:
        Dictionary containing raw data, reindexed data, forecasts, and combined data
    """
    eng = create_engine(
        f'{"postgresql"}://{os.environ.get("PG_USR", "")}:{os.environ.get("PG_PWD", "")}@{os.environ.get("PG_HOST", "")}:{os.environ.get("PG_PORT", "")}/{os.environ.get("PG_DB", "")}')
    if from_db:
        # Load the combined dataframe from the database
        logging.info("Loading combined data from database...")
        return pd.read_sql_table('combined_data', eng)
    else:
        # Initialize API clients and database connection
        logging.info("Initializing API clients and database connection")
        fred_ins = Fred(api_key=os.environ.get("FRED_API", ""))

        dfs = []

        logging.info("Getting data from World Bank")
        wb_data = fetch_world_bank_data(wb_indicators, "US")
        if wb_data:
            dfs.extend(wb_data)
            logging.info(f"Successfully fetched {len(wb_data)} datasets from World Bank")
        else:
            logging.warning("No data fetched from World Bank")

        logging.info("Getting data from FRED")
        fred_data = get_fred_data(fred_ins, fred_indicators)
        if fred_data:
            dfs.extend(fred_data)
            logging.info(f"Successfully fetched {len(fred_data)} datasets from FRED")
        else:
            logging.warning("No data fetched from FRED")

        logging.info("Getting data from Yahoo Finance")
        crypto_data = get_yahoo_data(yahoo_indicators, freq="1wk")
        if crypto_data:
            dfs.extend(crypto_data)
            logging.info(f"Successfully fetched {len(crypto_data)} datasets from Yahoo Finance")
        else:
            logging.warning("No data fetched from Yahoo Finance")

        if not dfs:
            logging.error("No data fetched from any source. Cannot proceed.")
            return None

        try:
            logging.info("Storing raw data in DB")
            store_data_in_db(eng, dfs)
            logging.info("Raw data stored successfully")
        except Exception as e:
            logging.error(f"Error storing raw data in DB: {str(e)}")
            # Continue processing even if DB storage fails

        logging.info("Generating ARIMA forecasts for next 10 years")
        forecasts = []
        for df in dfs:
            if df.empty:
                logging.warning(f"Skipping empty dataframe: {df.attrs.get('table_name', 'unknown')}")
                continue

            use_yearly_data = False
            table_name = df.attrs.get("table_name", "")

            # Use yearly data for World Bank data and FRED data except for 10y_treasury and nasdaq_com
            if ("world_bank" in table_name) or ("fred" in table_name and "10y_treasury" not in table_name and "nasdaq_com" not in table_name):
                use_yearly_data = True
                logging.info(f"Using yearly data for {table_name} predictions")

            # Analyze with auto_arima, using yearly data if appropriate
            results = analyze_with_auto_arima(df, use_yearly_data=use_yearly_data)
            if not results:
                logging.warning(f"No ARIMA results for {table_name}")
                continue

            # Make predictions, which will handle extrapolation from yearly to weekly if needed
            forecast = make_predictions(results, steps=10)  # 10 years of weekly data (52 weeks * 10 years)
            if forecast:
                forecasts.extend(forecast)
                logging.info(f"Successfully generated forecasts for {table_name}")
            else:
                logging.warning(f"No forecasts generated for {table_name}")

        # Check if we have any forecasts
        if not forecasts:
            logging.error("No forecasts generated. Cannot proceed.")
            return None

        try:
            logging.info("Storing forecasts data in DB")
            store_data_in_db(eng, forecasts)
            logging.info("Raw data stored successfully")
        except Exception as e:
            logging.error(f"Error storing forecasts data in DB: {str(e)}")

        reindexed_dfs = []
        reindexed_forecasts = []
        for df in dfs:
            reindexed_dfs.append(reindex_data([df], "2000-01-01", "2035-12-31", "W")[0])
        for forecast in forecasts:
            reindexed_forecasts.append(reindex_data([forecast], "2000-01-01", "2035-12-31", "W")[0])
        # Step 4: Combine all dataframes and forecasts into one final table
        logging.info("Combining all data and forecasts into one final table")
        combined_df = combine_dataframes(reindexed_dfs, reindexed_forecasts)

        # Print information about the combined dataframe to verify our changes
        logging.info(f"Combined dataframe shape: {combined_df.shape}")
        logging.info(f"Combined dataframe columns: {combined_df.columns.tolist()}")
        logging.info(f"Date range for combined dataframe: {combined_df.index.min()} to {combined_df.index.max()}")

        # Store processed data in DB
        try:
            logging.info("Storing processed data in DB")
            store_data_in_db(eng, [combined_df])
            logging.info("Processed data stored successfully")
        except Exception as e:
            logging.error(f"Error storing processed data in DB: {str(e)}")
            # Continue even if DB storage fails

        return {
            "raw_data": dfs,
            "forecasts": forecasts,
            "reindexed_data": reindexed_dfs,
            "reindexed_forecasts": reindexed_forecasts,
            "combined_data": combined_df
        }

def main():

    result = get_data(False)
    combined_data = result["combined_data"]
    logging.info(f"Combined data shape: {combined_data.shape}")
    logging.info(f"Combined data columns: {combined_data.columns.tolist()}")

    # Check if forecasts are included in the output
    current_date = pd.Timestamp.now()
    future_data = combined_data[combined_data.index > current_date]
    logging.info(f"Future data shape: {future_data.shape}")
    logging.info(f"Future data date range: {future_data.index.min()} to {future_data.index.max()}")

    # Print a sample of the future data to verify forecasts are included
    logging.debug(f"Sample of future data:\n{future_data.head(10)}")

    # Print the tail of the combined data for reference
    logging.debug(f"Tail of combined data:\n{combined_data.tail(10)}")

if __name__ == "__main__":
    main()
