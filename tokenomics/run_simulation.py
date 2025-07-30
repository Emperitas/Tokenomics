"""
Simulation runner for blockchain token value model.
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred

# Add the parent directory to the path so we can import the tokenomics package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.etl import analyze_with_auto_arima, make_predictions, get_fred_data
from tokenomics.enviroment.token_market_model import TokenMarketModel
import yaml
from pathlib import Path



def run_single_simulation(config: dict):
    """
    Run a single simulation with the given parameters.
    
    Args:
        config: Dictionary containing simulation parameters
        
    Returns:
        model: The final model state
        data: DataFrame containing the collected data
    """
    # Set random seed if provided
    if config.get("seed") is not None:
        np.random.seed(config["seed"])
        random.seed(config["seed"])
    config["token_supply_per_tick"] = config["forecasts"]["token_schedule"]/config["token_reduction_ratio"]
    config["max_token_count"] = config["max_token_count"]/config["token_reduction_ratio"]

    # Create model
    model = TokenMarketModel(
        initial_price=config["initial_price"],
        initial_hodlers=config["initial_hodlers"],
        initial_speculators=config["initial_speculators"],
        initial_immediate_users=config["initial_immediate_users"],
        initial_delayed_users=config["initial_delayed_users"],
        initial_tokens_per_agent=config["initial_tokens_per_agent"],
        initial_dollars_per_agent=config["initial_dollars_per_agent"],
        static_agent_distribution=config["static_agent_distribution"],
        max_token_count=config["max_token_count"],
        max_steps=config["max_steps"],
        token_supply_per_tick=config["token_supply_per_tick"],
        mean_agent_inflow=config["mean_agent_inflow"],
        std_agent_inflow=config["std_agent_inflow"],
        macro_data=config["forecasts"],
        config=config
    )
    
    # Run model
    for i in range(config["max_steps"]):
        model.step()
    
    # Get data
    data = model.datacollector.get_model_vars_dataframe()
    agent_data = None # model.datacollector.get_agent_vars_dataframe()
    return model, data, agent_data


def visualize_results(data, title="Token Market Simulation Results", filename="fig.png"):
    """
    Visualize simulation results.
    
    Args:
        data: DataFrame containing the simulation data
        title: Title for the plots
    """
    # Create figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # Plot price
    axs[0, 0].plot(data.index, data["Price"])
    axs[0, 0].set_title("Token Price")
    axs[0, 0].set_xlabel("Step")
    axs[0, 0].set_ylabel("Price")
    axs[0, 0].grid(True)
    
    # Plot active agents
    axs[0, 1].plot(data.index, data["Active Agents"])
    axs[0, 1].set_title("Active Agents")
    axs[0, 1].set_xlabel("Step")
    axs[0, 1].set_ylabel("Count")
    axs[0, 1].grid(True)
    
    # Plot agent types
    agent_types = ["Hodlers", "Speculators", "Immediate Users", "Delayed Users", "Node Operators"]
    for i, agent_type in enumerate(agent_types):
        if agent_type in data.columns:
            axs[1, 0].plot(data.index, data[agent_type], label=agent_type)
    axs[1, 0].set_title("Agent Types")
    axs[1, 0].set_xlabel("Step")
    axs[1, 0].set_ylabel("Count")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Plot total tokens and dollars
    axs[1, 1].plot(data.index, data["Total Tokens"], label="Total Tokens")
    axs[1, 1].set_title("Total Tokens")
    axs[1, 1].set_xlabel("Step")
    axs[1, 1].set_ylabel("Count")
    axs[1, 1].grid(True)
    
    ax2 = axs[1, 1].twinx()
    ax2.plot(data.index, data["Total Dollars"], 'r-', label="Total Dollars")
    ax2.set_ylabel("Dollars")
    
    lines1, labels1 = axs[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs[1, 1].legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    
    # Plot staked tokens
    if "Staked Tokens" in data.columns:
        axs[2, 0].plot(data.index, data["Staked Tokens"])
        axs[2, 0].set_title("Staked Tokens")
        axs[2, 0].set_xlabel("Step")
        axs[2, 0].set_ylabel("Count")
        axs[2, 0].grid(True)
    
    # Plot price change
    if len(data) > 1:
        price_changes = data["Price"].pct_change().fillna(0)
        axs[2, 1].plot(data.index[1:], price_changes[1:])
        axs[2, 1].set_title("Price Change (%)")
        axs[2, 1].set_xlabel("Step")
        axs[2, 1].set_ylabel("Percent Change")
        axs[2, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    plt.show()



def save_results(data, filename="simulation_results_token_schedule3.csv"):
    """
    Save simulation results to a CSV file.
    
    Args:
        data: DataFrame containing the simulation data
        filename: Name of the file to save
    """
    data.to_csv(filename)
    print(f"Results saved to {filename}")


def get_macro_data():
    """
    Retrieve macroeconomic data from the Federal Reserve Economic Database (FRED).
    
    This function fetches key economic indicators including CPI, federal funds rate,
    GDP, 10-year treasury yield, and NASDAQ composite index.
    
    Returns:
        List[pd.DataFrame]: A list of pandas DataFrames containing the fetched economic data
    """
    fred_indicators = {
        "cpi": "CPIAUCSL",
        "federal_rate": "FEDFUNDS",
        "gdp": "GDP",
        "10y_treasury": "DGS10",
        "nasdaq_com": "NASDAQCOM",
    }

    fred_ins = Fred(api_key=os.environ.get("FRED_API", ""))
    fred_data = get_fred_data(fred_ins, fred_indicators)
    return fred_data

def run_predictions(fred_data):
    """
    Generate forecasts for economic indicators using ARIMA models.
    
    This function analyzes the provided economic data using auto ARIMA models
    and generates forecasts for different time periods based on the data frequency.
    It handles different data frequencies (daily, weekly, monthly, quarterly, yearly)
    and resamples the forecasts to a consistent frequency when needed.
    
    Args:
        fred_data (List[pd.DataFrame]): List of pandas DataFrames containing economic data
        
    Returns:
        dict: Dictionary mapping indicator names to forecast DataFrames
    """
    forecasts = []
    results = {}
    for df in fred_data:
        if df.empty:
            continue

        use_yearly_data = False
        table_name = df.attrs.get("table_name", "")

        # Analyze with auto_arima, using yearly data if appropriate
        results = {**results, **analyze_with_auto_arima(df, use_yearly_data=use_yearly_data)}
        if not results:
            continue

    final_forecast = {}
    for name, result in results.items():
        s = 10
        model = result["model"]
        step_length = pd.DataFrame(model.predict(steps=2))
        diff = (step_length.index[2] - step_length.index[1])
        if isinstance(diff, pd.Timedelta):
            diff = diff.days
        if diff < 7:
            print('daily')
            forecast_values = pd.DataFrame(model.predict(n_periods=s * 365))
            indexresampled = np.linspace(forecast_values.index[0], forecast_values.index[-1], num=s * 52, endpoint=True)
            indexresampled = pd.Index(indexresampled)
            df_resampled = forecast_values.reindex(df.index.union(indexresampled)).interpolate().loc[indexresampled]
            final_forecast[name] = df_resampled
        elif 6 < diff < 8:
            print('weekly')
            forecast_values = pd.DataFrame(model.predict(n_periods=s * 52))
            final_forecast[name] = forecast_values
        elif 25 < diff < 35:
            print('monthly')
            forecast_values = pd.DataFrame(model.predict(n_periods=s * 12))
            forecast_values = forecast_values.resample("W").interpolate()
            final_forecast[name] = forecast_values
        elif 35 < diff < 100:
            print('quarterly')
            forecast_values = pd.DataFrame(model.predict(n_periods=s * 4))
            forecast_values = forecast_values.resample("W").interpolate()
            final_forecast[name] = forecast_values
        else:
            print('yearly')
            forecast_values = pd.DataFrame(model.predict(n_periods=s))
            forecast_values = forecast_values.resample("W").interpolate()
            final_forecast[name] = forecast_values

    return final_forecast


def transform_predictions(data):
    """
    Transform prediction data into a standardized format for simulation.
    
    This function takes the raw prediction data from various sources and transforms
    it into a standardized format with consistent columns and indices. It handles
    different economic indicators (CPI, treasury yields, NASDAQ, GDP, federal rate)
    and ensures they are properly aligned and formatted for use in the simulation.
    
    Args:
        data (dict): Dictionary containing prediction data for different indicators
        
    Returns:
        pd.DataFrame: Transformed prediction data with standardized format
    """
    prediction_data = pd.concat(
        [data["fred_cpi"], pd.DataFrame([416.208670, 416.208670], index=['2035-06-10', '2035-06-17'])])
    prediction_data.columns = ["cpi"]
    prediction_data["y10_treasury"] = data["fred_10y_treasury"].values
    prediction_data["nasdaq_com"] = data["fred_nasdaq_com"].values
    prediction_data["gdp"] = pd.concat([data["fred_gdp"], pd.DataFrame([[] for _ in range(10)])],
                                       axis=1).ffill().values
    prediction_data["federal_rate"] = pd.concat(
        [data["fred_federal_rate"], pd.DataFrame([[] for _ in range(2)])], axis=1).ffill().values
    prediction_data.index = pd.to_datetime(prediction_data.index)
    
    return prediction_data


def add_token_schedule(data, path:str = "./Xeta Estimated Token Distribution Schedule.csv"):
    """
    Add token distribution schedule to the simulation data.
    
    This function reads a token distribution schedule from a CSV file and adds it
    to the simulation data. It processes the schedule to extract the monthly token
    distribution amounts and adds them to the data at the appropriate time points.
    It also calculates percentage increases for other economic indicators.
    
    Args:
        data (pd.DataFrame): DataFrame containing simulation data
        path (str): Path to the CSV file containing the token distribution schedule
        
    Returns:
        pd.DataFrame: Processed data with token schedule and percentage increases
    """
    schedule = pd.read_csv(path)
    schedule.columns = ["year", "month", "original_token_per_month", "original_acc_token", "new_token_per_month", "new_acc_token"]
    schedule = schedule.drop(columns=["original_token_per_month", "original_acc_token", "new_acc_token"])
    schedule = pd.to_numeric(schedule["new_token_per_month"].str.replace(",", ""))
    data["token_schedule"] = 0

    c_month = 0
    row_index = 0
    for i, row in data.iterrows():
        if i.month != c_month:
            c_month = i.month
            data.at[i, "token_schedule"] = schedule.iloc[row_index]
            row_index += 1
        else:
            data.at[i, "token_schedule"] = 0

    for c in data.columns:
        if c != "token_schedule":
            data[f"{c}_increase"] = data[c].pct_change()
    model_data = data.drop(["cpi", "y10_treasury", "nasdaq_com", "gdp", "federal_rate"], axis=1).fillna(
        0).to_csv("model_data.csv", index=False)
    return model_data


def main():
    """
    Main function to run the simulation.
    """
    forecasts = pd.read_csv("../data/model_data.csv")
    print("Running blockchain token value simulation...")
    conf = yaml.safe_load(Path('config/parameters.yaml').read_text())
    # Run a single simulation
    conf["forecasts"] = forecasts
    model, data , agent_data = run_single_simulation(conf)
    
    # Visualize results
    visualize_results(data, filename= f'token_market_simulation_tr{conf["token_reduction_ratio"]}_m{conf["mean_agent_inflow"]}_s{conf["std_agent_inflow"]}agents_{str(conf["static_agent_distribution"]).replace(","," ")}.png')


    # Save results
    save_results(data, f'model_results_tr{conf["token_reduction_ratio"]}_m{conf["mean_agent_inflow"]}_s{conf["std_agent_inflow"]}_agents{str(conf["static_agent_distribution"]).replace(","," ")}.csv')
    #save_results(agent_data, f'agent_results_tr{conf["token_reduction_ratio"]}_m{conf["mean_agent_inflow"]}_s{conf["std_agent_inflow"]}.csv')
    
    print("Simulation complete.")


if __name__ == "__main__":
    main()