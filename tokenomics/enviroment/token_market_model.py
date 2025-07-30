"""
Token market model implementation using MESA.
"""
from mesa import Model, Agent
from mesa.datacollection import DataCollector
import numpy as np
import pandas as pd
import random
import threading
import concurrent.futures
from functools import partial

from tokenomics.agent.mesa_agents import (
    BaseAgent, HodlerAgent, SpeculatorAgent,
    ImmediateUserAgent, DelayedUserAgent,
    EnvironmentAgent
)

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class Transaction:
    type: str
    amount: float
    price: float
    total: float
    step: int



class Order:
    """
    Represents an order in the token market.
    """

    def __init__(self, agent:Agent, order_type:str, price:float, amount:int):
        """
        Initialize a new Order.

        Args:
            agent: The agent placing the order
            order_type: 'buy' or 'sell'
            price: Price per token
            amount: Number of tokens
        """
        self.agent = agent
        self.order_type = order_type
        self.price = price
        self.amount = amount
        self.timestamp = 0  # Will be set when added to the order book

    def __str__(self):
        return f"{self.order_type.upper()} {self.amount} @ {self.price} by {self.agent.unique_id}"


class OrderBook:
    """
    Represents an order book for the token market.
    """

    def __init__(self):
        """
        Initialize a new OrderBook.
        """
        self.buy_orders = []  # List of buy orders
        self.sell_orders = []  # List of sell orders
        self.timestamp = 0  # Current timestamp

    def add_order(self, order):
        """
        Add an order to the order book.

        Args:
            order: The order to add
        """
        order.timestamp = self.timestamp
        self.timestamp += 1

        if order.order_type == 'buy':
            self.buy_orders.append(order)
            # Sort buy orders by price (descending) and timestamp (ascending)
            self.buy_orders.sort(key=lambda x: (-x.price, x.timestamp))
        else:  # sell order
            self.sell_orders.append(order)
            # Sort sell orders by price (ascending) and timestamp (ascending)
            self.sell_orders.sort(key=lambda x: (x.price, x.timestamp))

    def remove_order(self, order):
        """
        Remove an order from the order book.

        Args:
            order: The order to remove
        """
        if order.order_type == 'buy':
            self.buy_orders.remove(order)
        else:  # sell order
            self.sell_orders.remove(order)

    def get_best_buy_price(self):
        """
        Get the best (highest) buy price.

        Returns:
            float: The best buy price, or 0 if no buy orders
        """
        if self.buy_orders:
            return self.buy_orders[0].price
        return 0

    def get_best_sell_price(self):
        """
        Get the best (lowest) sell price.

        Returns:
            float: The best sell price, or float('inf') if no sell orders
        """
        if self.sell_orders:
            return self.sell_orders[0].price
        return float('inf')

    def get_market_price(self):
        """
        Get the current market price (midpoint between best buy and sell).

        Returns:
            float: The current market price, or None if no orders
        """
        best_buy = self.get_best_buy_price()
        best_sell = self.get_best_sell_price()

        if best_buy > 0 and best_sell < float('inf'):
            return (best_buy + best_sell) / 2
        elif best_buy > 0:
            return best_buy
        elif best_sell < float('inf'):
            return best_sell
        else:
            return None


class TokenMarketModel(Model):
    """
    Token market model for blockchain token value simulation.
    """

    def __init__(
            self,
            initial_price: float = 100.0,
            initial_hodlers: int = 10,
            initial_speculators: int = 10,
            initial_immediate_users: int = 10,
            initial_delayed_users: int = 10,
            initial_tokens_per_agent: int = 10,
            initial_dollars_per_agent: int = 1000,
            static_agent_distribution: list = [],
            mean_agent_inflow: int = 100,
            std_agent_inflow: int = 10,
            max_token_count: int = 100000,
            max_steps: int = 100,
            token_supply_per_tick: int = 13333,
            macro_data: pd.DataFrame = None,
            config:dict = None,
    ):
        """
        Initialize a new TokenMarketModel.

        Args:
            initial_price: Initial token price
            initial_hodlers: Initial number of hodler agents
            initial_speculators: Initial number of speculator agents
            initial_immediate_users: Initial number of immediate user agents
            initial_delayed_users: Initial number of delayed user agents
            initial_tokens_per_agent: Initial number of tokens per agent
            initial_dollars_per_agent: Initial amount of dollars per agent
            max_token_count: Maximum number of tokens in the token supply
            max_steps: Maximum number of steps to run the simulation
        """
        super().__init__()
        self.initial_price = initial_price
        self.current_price = initial_price
        self.price_history = [initial_price]
        self.volume_history = []
        self.max_steps = max_steps
        self.mean_agent_inflow = mean_agent_inflow,
        self.std_agent_inflow = std_agent_inflow,
        self.macro_data = macro_data
        self.config = config
        self.initial_dollars_for_agent_lbound = 100
        self.initial_dollars_for_agent_ubound = 10000
        self.fixed_sub = 10
        self.mobile_sub = 2
        self.static_agent_distribution = static_agent_distribution

        # Initialize schedule
        self.agents.shuffle_do("step")

        # Initialize order book
        self.order_book = OrderBook()

        # Initialize environment agent
        self.environment_agent = None

        # Initialize agents
        self.initialize_agents(
            initial_hodlers,
            initial_speculators,
            initial_immediate_users,
            initial_delayed_users,
            initial_tokens_per_agent,
            initial_dollars_per_agent,
            max_token_count,
            token_supply_per_tick,
        )

        # Initialize data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Price": lambda m: m.current_price,
                "Total Tokens": lambda m: sum(a.tokens_held for a in m.agents),
                "Total Dollars": lambda m: sum(a.dollars_held for a in m.agents),
                "Active Agents": lambda m: sum(1 for a in m.agents if a.active),
                "Hodlers": lambda m: sum(
                    1 for a in m.agents
                    if (isinstance(a, HodlerAgent) and a.active)
                ),
                "Speculators": lambda m: sum(
                    1 for a in m.agents
                    if (isinstance(a, SpeculatorAgent) and a.active)
                ),
                "Immediate Users": lambda m: sum(
                    1 for a in m.agents
                    if (isinstance(a, ImmediateUserAgent) and a.active)
                ),
                "Delayed Users": lambda m: sum(
                    1 for a in m.agents
                    if (isinstance(a, DelayedUserAgent) and a.active)
                ),
                "Token Momentum": lambda m: m.current_price / m.price_history[-2] if len(m.price_history) >= 2 else 0,
                "Token Velocity": lambda m: m.volume_history[-1] if m.volume_history else 0,
                # Environment agent metrics
                #"Burned Tokens": lambda m: m.environment_agent.burned_tokens if m.environment_agent else 0,
                #"Granted Tokens": lambda m: m.environment_agent.granted_tokens if m.environment_agent else 0,
                "Received Tokens": lambda m: m.environment_agent.received_tokens if m.environment_agent else 0,
                "Sold Tokens": lambda m: m.environment_agent.sold_tokens if m.environment_agent else 0,
                "Tokens Held by Environment Agent": lambda m: m.environment_agent.tokens_held if m.environment_agent else 0,

            },
            # agent_reporters={
            #     "Tokens": lambda a: a.tokens_held,
            #     "Dollars": lambda a: a.dollars_held,
            #     "Active": lambda a: a.active,
            #     "Type": lambda a: type(a).__name__,
            # }
        )

    def initialize_agents(
            self,
            initial_hodlers,
            initial_speculators,
            initial_immediate_users,
            initial_delayed_users,
            initial_tokens_per_agent,
            initial_dollars_per_agent,
            max_token_count,
            token_supply_per_tick,
    ):
        """
        Initialize agents for the simulation.

        Args:
            initial_hodlers: Number of hodler agents to create
            initial_speculators: Number of speculator agents to create
            initial_immediate_users: Number of immediate user agents to create
            initial_delayed_users: Number of delayed user agents to create
            initial_tokens_per_agent: Initial number of tokens per agent
            initial_dollars_per_agent: Initial amount of dollars per agent
            max_token_count: Maximum number of tokens in the token supply
            token_supply_per_tick: Number of tokens to sell on market per tick
        """
        # Create environment agent first
        self.environment_agent = EnvironmentAgent(
            self,
            tokens_held=0,
            dollars_held=0,
            max_tokens=max_token_count,
            token_supply_per_tick=token_supply_per_tick
        )
        self.agents.add(self.environment_agent)
        # Create hodler agents
        for i in range(initial_hodlers):
            agent = HodlerAgent(
                self,
                tokens_held=initial_tokens_per_agent,
                dollars_held=initial_dollars_per_agent,
                sell_threshold=random.uniform(0.05, 0.3)
            )
            self.agents.add(agent)

        # Create speculator agents
        for i in range(initial_speculators):
            agent = SpeculatorAgent(
                self,
                tokens_held=initial_tokens_per_agent,
                dollars_held=initial_dollars_per_agent,
                risk_tolerance=random.uniform(0.2, 0.8)
            )
            self.agents.add(agent)

        # Create immediate user agents
        for i in range(initial_immediate_users):
            agent = ImmediateUserAgent(
                self,
                tokens_held=initial_tokens_per_agent,
                dollars_held=initial_dollars_per_agent,
                usage_rate=random.uniform(0.5, 1.0)
            )
            self.agents.add(agent)

        # Create delayed user agents
        for i in range(initial_delayed_users):
            agent = DelayedUserAgent(
                self,
                tokens_held=initial_tokens_per_agent,
                dollars_held=initial_dollars_per_agent,
                future_usage_time=random.randint(2, 10),
                price_expectation=random.uniform(1.05, 1.2)
            )
            self.agents.add(agent)

    def agent_inflow_type_distribution(self):
        """
        Calculate the distribution of agent types for new agent inflow.
        
        This method determines the proportion of each agent type (hodler, speculator,
        immediate user, delayed user) for new agents entering the simulation. It uses
        the static agent distribution for the first few steps, then adjusts based on
        market conditions.
        
        Returns:
            List[float]: A list of proportions for each agent type, normalized to sum to 1
        """
        if self.steps < 5:
            return self.static_agent_distribution
        else:
            hodler_inflow = self.static_agent_distribution[0]
            speculator_inflow = self.static_agent_distribution[1] # * (1.0 + self.get_market_trend()/10.0-0.02) if self.get_market_trend() else 0.0
            immediate_user_inflow = self.static_agent_distribution[2] #* 2.0/self.current_price
            delayed_user_inflow = self.static_agent_distribution[3] #* 2.0/self.current_price
            total_distribution = hodler_inflow + speculator_inflow + immediate_user_inflow + delayed_user_inflow
            return [hodler_inflow/total_distribution, speculator_inflow/total_distribution, immediate_user_inflow/total_distribution, delayed_user_inflow/total_distribution]

    def add_new_agents(self):
        """
        Add new agents to the simulation based on market conditions.
        
        This method determines the number and types of new agents to add to the simulation
        at each step. It adjusts the number of agents based on market trends and randomly
        selects agent types according to the distribution returned by agent_inflow_type_distribution.
        It also updates economic parameters like initial dollars and subscription prices based
        on inflation and GDP growth.
        
        The method creates agents of different types (Hodler, Speculator, Immediate User, Delayed User)
        with appropriate initial parameters and adds them to the simulation.
        """
        if isinstance(self.mean_agent_inflow, tuple):
            self.mean_agent_inflow = self.mean_agent_inflow[0]
        if isinstance(self.std_agent_inflow, tuple):
            self.std_agent_inflow = self.std_agent_inflow[0]
        mean = self.mean_agent_inflow * (1.0 + (self.get_market_trend()/10.0 if self.get_market_trend() else 0.0))
        std = self.std_agent_inflow * (1.0 + np.random.normal(0.0, 0.1))
        self.std_agent_inflow = std
        print(f"mean_agent_inflow {self.mean_agent_inflow}")
        print(f"std_agent_inflow {self.std_agent_inflow}")
        number_of_new_agents = int(max(0, np.random.normal(mean, std)))
        self.mean_agent_inflow = number_of_new_agents
        agent_distribution = self.agent_inflow_type_distribution()
        step = self.steps
        self.initial_dollars_for_agent_lbound = int(self.initial_dollars_for_agent_lbound * (1.0 + self.macro_data["cpi_increase"][step-1] + self.macro_data["gdp_increase"][step-1]))
        self.initial_dollars_for_agent_ubound = int(self.initial_dollars_for_agent_ubound * (
                    1.0 + self.macro_data["cpi_increase"][step-1] + self.macro_data["gdp_increase"][step-1]))
        self.fixed_sub = round(self.fixed_sub * (1.0 + self.macro_data["cpi_increase"][step-1] + self.macro_data["gdp_increase"][step-1]), 2)
        self.mobile_sub = round(self.mobile_sub * (1.0 + self.macro_data["cpi_increase"][step-1] + self.macro_data["gdp_increase"][step-1]), 2)
        agent_types = random.choices(
            population=['HodlerAgent', 'SpeculatorAgent', 'ImmediateUserAgent', 'DelayedUserAgent'],
            weights=agent_distribution, k=number_of_new_agents)
        for i in agent_types:
            if i == 'HodlerAgent':
                agent = HodlerAgent(
                    self,
                    tokens_held=0,
                    dollars_held=random.randint(self.initial_dollars_for_agent_lbound, self.initial_dollars_for_agent_ubound),
                )
            elif i == "SpeculatorAgent":
                agent = SpeculatorAgent(
                    self,
                    tokens_held=0,
                    dollars_held=random.randint(self.initial_dollars_for_agent_lbound, self.initial_dollars_for_agent_ubound),
                    risk_tolerance=random.uniform(0.2, 0.8)
                )
            elif i == "ImmediateUserAgent":
                agent = ImmediateUserAgent(
                    self,
                    tokens_held=0,
                    dollars_held=random.randint(self.initial_dollars_for_agent_lbound, self.initial_dollars_for_agent_ubound),
                    usage_rate=random.uniform(0.5, 1.0),
                    max_token_price=np.random.choice([self.mobile_sub, self.fixed_sub],1,p=[0.8,0.2])
                )
            elif i == "DelayedUserAgent":
                agent = DelayedUserAgent(
                    self,
                    tokens_held=0,
                    dollars_held=random.randint(self.initial_dollars_for_agent_lbound, self.initial_dollars_for_agent_ubound),
                    future_usage_time=random.randint(2, 10),
                    price_expectation=random.uniform(1.05, 1.5),
                    max_token_price = np.random.choice([self.mobile_sub, self.fixed_sub], 1, p=[0.8, 0.2])
                )
            else:
                continue
            self.agents.add(agent)

    def step(self):
        """
        Advance the model by one step.
        """
        # Collect data
        print(f"step {self.steps}")
        self.datacollector.collect(self)
        # Add new agents to simulation
        self.add_new_agents()
        # Step all agents using batch processing for better performance
        self.batch_process_agents()
        # Process orders
        self.process_orders()
        # Update market price
        self.update_market_price()
        # Check if simulation should end
        if self.steps >= self.max_steps:
            self.running = False
        print(f"token price {self.current_price}")


    def batch_process_agents(self):
        """
        Process all agents in batches for better performance when scaling up.
        This replaces the sequential processing of agents with a more efficient approach.
        """
        # Get current market data once for all agents
        current_price = self.get_current_price()
        market_trend = self.get_market_trend()
        market_volatility = self.get_market_volatility()

        # Prepare market data dictionary that all agents will use
        market_data = {
            'price': current_price,
            'trend': market_trend,
            'volatility': market_volatility,
            'step': self.steps,
            'cpi_increase': self.macro_data['cpi_increase'][self.steps],
        }

        active_agents = [agent for agent in self.agents if agent.active]
        # Process agents by type for more efficient batch operations
        self._batch_process_agent_type(active_agents, market_data)

    def _process_single_agent(self, agent, market_data):
        """
        Process a single agent with the given market data.

        Args:
            agent: The agent to process
            market_data: Dictionary containing current market data
        """
        # Set market observations directly instead of calling observe_market()
        agent.market_observations = market_data.copy()
        agent.make_decisions()
        return agent

    def _batch_process_agent_type(self, agents, market_data):
        """
        Process all agents of a specific type in batch using threading.

        Args:
            agents: List of active agents
            market_data: Dictionary containing current market data
        """
        if not agents:
            return

        # Use threading to process agents in parallel
        num_threads = min(threading.active_count(), len(agents))

        if num_threads > 1 and len(agents) > 10:  # Only use threading if there are enough agents
            # Create a partial function with the market_data argument preset
            process_agent = partial(self._process_single_agent, market_data=market_data)

            # Create a thread pool executor
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Process agents in parallel
                executor.map(process_agent, agents)
        else:
            # If only one thread is available or there are very few agents, process sequentially
            for agent in agents:
                self._process_single_agent(agent, market_data)

    def process_orders(self):
        """
        Process orders in the order book.
        """
        # Match orders
        self.match_orders()

        # Clear expired orders
        self.clear_expired_orders()

    def match_orders(self):
        """
        Match buy and sell orders using a single clearing price
        that maximizes the number of matched orders in the auction.
        """
        buy_orders = self.order_book.buy_orders
        sell_orders = self.order_book.sell_orders

        if not buy_orders or not sell_orders:
            return

        # Use set comprehension for unique prices
        possible_prices = sorted({order.price for order in buy_orders} |
                                 {order.price for order in sell_orders})

        # Track max quantity and price using max() with key function
        def get_matched_quantity(price):
            buy_quantity = sum(order.amount for order in buy_orders if order.price >= price)
            sell_quantity = sum(order.amount for order in sell_orders if order.price <= price)
            return min(buy_quantity, sell_quantity)

        # Find optimal clearing price and quantity in one pass
        clearing_price, max_quantity = max(
            ((price, get_matched_quantity(price)) for price in possible_prices),
            key=lambda x: (round(x[1], 2), -x[0]),  # Sort by quantity first, then lowest price
            default=(0, 0)
        )
        self.volume_history.append(max_quantity)
        if clearing_price > 0 and max_quantity > 0:
            self.execute_auction_trades(clearing_price)

    def execute_auction_trades(self, clearing_price):
        """
        Execute all trades at the specified clearing price using batch processing.
        """
        buy_orders = self.order_book.buy_orders
        sell_orders = self.order_book.sell_orders

        while buy_orders and sell_orders:
            buy_order = buy_orders[0]
            sell_order = sell_orders[0]

            # Skip orders that don't match the clearing price, but don't break the loop
            if buy_order.price < clearing_price:
                self.order_book.remove_order(buy_order)
                continue
            if sell_order.price > clearing_price:
                self.order_book.remove_order(sell_order)
                continue

            trade_amount = min(buy_order.amount, sell_order.amount)
            self.execute_trade(buy_order.agent, sell_order.agent, trade_amount, clearing_price)

            # Update orders and remove if completed
            buy_order.amount -= trade_amount
            sell_order.amount -= trade_amount

            if buy_order.amount <= 0:
                self.order_book.remove_order(buy_order)
            if sell_order.amount <= 0:
                self.order_book.remove_order(sell_order)

        self.current_price = clearing_price
        self.price_history.append(clearing_price)

    def execute_trade(self, buyer, seller, amount, price):
        """
        Execute a trade between two agents.

        Args:
            buyer: The buying agent
            seller: The selling agent
            amount: Number of tokens to trade
            price: Price per token
        """
        # Calculate total cost
        total_cost = amount * price

        # Transfer tokens to buyer (dollars were already reserved when order was placed)
        buyer.tokens_held += amount

        # Transfer dollars to seller (tokens were already reserved when order was placed)
        seller.dollars_held += total_cost

        # Record transaction in agents' history
        buyer.transaction_history.append({
            'type': 'buy',
            'amount': amount,
            'price': price,
            'total_cost': total_cost,
            'step': self.steps
        })

        seller.transaction_history.append({
            'type': 'sell',
            'amount': amount,
            'price': price,
            'total_revenue': total_cost,
            'step': self.steps
        })

        # Update current price


    def clear_expired_orders(self):
        """
        Clear expired orders from the order book and return reserved resources to agents.
        """
        # Set a maximum order age (in steps)
        max_order_age = 2
        current_timestamp = self.order_book.timestamp

        # Check buy orders
        expired_buy_orders = []
        for order in self.order_book.buy_orders:
            if current_timestamp - order.timestamp > max_order_age:
                expired_buy_orders.append(order)

                # Return reserved dollars to the agent
                cost = order.amount * order.price
                order.agent.dollars_held += cost

        # Check sell orders
        expired_sell_orders = []
        for order in self.order_book.sell_orders:
            if current_timestamp - order.timestamp > max_order_age:
                expired_sell_orders.append(order)

                # Return reserved tokens to the agent
                order.agent.tokens_held += order.amount

        # Remove expired orders from the order book
        for order in expired_buy_orders:
            self.order_book.remove_order(order)

        for order in expired_sell_orders:
            self.order_book.remove_order(order)

    def place_order(self, agent, order_type, price, amount):
        """
        Place an order in the order book.

        Args:
            agent: The agent placing the order
            order_type: 'buy' or 'sell'
            price: Price per token
            amount: Number of tokens

        Returns:
            Order: The created order
        """
        # Create a new order
        order = Order(agent, order_type, price, amount)

        # Add the order to the order book
        self.order_book.add_order(order)

        return order

    def update_market_price(self):
        """
        Update the current market price.
        """
        market_price = self.order_book.get_market_price()
        if market_price is not None:
            self.current_price = market_price
            self.price_history.append(market_price)

    def get_current_price(self):
        """
        Get the current token price.

        Returns:
            float: The current token price
        """
        return self.current_price

    def get_market_trend(self):
        """
        Calculate the market trend based on recent price history.

        Returns:
            float: The market trend (-1 to 1)
        """
        if len(self.price_history) < 2:
            return 0

        # Calculate trend over the last 10 steps (or fewer if not available)
        history_length = min(10, len(self.price_history))
        recent_prices = self.price_history[-history_length:]

        if recent_prices[0] == 0:
            return 0

        # Calculate percentage change
        percent_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

        # Normalize to -1 to 1 range
        return max(-1, min(1, percent_change))

    def get_market_volatility(self):
        """
        Calculate the market volatility based on recent price history.

        Returns:
            float: The market volatility (0 to 1)
        """
        if len(self.price_history) < 2:
            return 0

        # Calculate volatility over the last 10 steps (or fewer if not available)
        history_length = min(10, len(self.price_history))
        recent_prices = self.price_history[-history_length:]

        # Calculate standard deviation of percentage changes
        changes = [
            (recent_prices[i] - recent_prices[i - 1]) / recent_prices[i - 1]
            for i in range(1, len(recent_prices))
        ]

        if not changes:
            return 0

        std_dev = np.std(changes)

        # Normalize to 0 to 1 range (assuming max volatility is 0.2 or 20%)
        return min(1, std_dev / 0.2)
