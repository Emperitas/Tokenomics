# XETA-tokenomics: Blockchain Token Value Simulation

An Agent-Based Model for simulating blockchain token value dynamics using the MESA Python framework.

## Overview

This project implements an agent-based model to simulate the dynamics of blockchain token value. The model includes various types of agents with different behaviors, a token market for exchange, and data collection for analysis.

The simulation aims to model how different agent behaviors affect token value over time, providing insights into the factors that influence blockchain token economics.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/Emperitas/Tokenomics.git
   cd tokenomics
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On Unix or MacOS
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Agent Types

The model includes the following agent types:

1. **Holder**: People who buy believing that the long-term trajectory is upward indefinitely. They buy and hold tokens indefinitely.
   - Actions: Observation, enter and immediately buy, liquidate (sell and immediately exit)

2. **Speculator**: People hoping to make money arbitraging the market.
   - Actions: Observation, enter and immediately buy, buy, sell, liquidate (sell and immediately exit)

3. **Immediate User**: People who want to use the token immediately.
   - Actions: Observation, enter and immediately buy, immediately spend, exit

4. **Delayed User**: People who want to use the token in the future and are purchasing now because they anticipate higher future costs.
   - Actions: Observation, purchase for future use, use tokens at a later time

## Usage

### Running a Single Simulation

To run a single simulation with default parameters:

```python
python -m tokenomics.run_simulation
```

### Customizing Simulation Parameters

You can customize the simulation by modifying the parameters in the `config/parameters.yaml` file.

## Data Collection and Visualization

The model collects data on various metrics, including:

- Token price
- Number of active agents by type
- Total tokens and dollars in the system
- Staked tokens
- Price changes

The `visualize_results` function creates plots for these metrics, providing insights into the dynamics of the token market.

## Project Structure

- `tokenomics/agent/mesa_agents.py`: Implementation of agent types
- `tokenomics/enviroment/token_market_model.py`: Implementation of the token market model
- `tokenomics/run_simulation.py`: Simulation runner and visualization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
