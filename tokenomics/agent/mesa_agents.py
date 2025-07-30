"""
Mesa-based agent implementations for blockchain token value simulation.
"""
from mesa import Agent
import random
import numpy as np


class BaseAgent(Agent):
    """
    Base agent class for all agent types in the blockchain token value simulation.
    Extends Mesa's Agent class with common attributes and methods.
    """
    def __init__(self, model, tokens_held:int=0, dollars_held:float=0):
        """
        Initialize a new BaseAgent.

        Args:
            model: The model instance the agent belongs to
            tokens_held: Initial number of tokens held by the agent
            dollars_held: Initial amount of dollars held by the agent
        """
        super().__init__(model)
        self.tokens_held = tokens_held
        self.dollars_held = dollars_held
        self.active = True  # Whether the agent is active in the simulation
        self.transaction_history = []  # List of transactions made by the agent
        self.market_observations = {}  # Store market observations

    def step(self):
        """
        Base step method to be overridden by subclasses.
        This method is called for each agent at each step of the simulation.
        """
        if not self.active:
            return

        # Observe market conditions
        self.observe_market()

        # Make decisions based on observations
        self.make_decisions()

    def observe_market(self):
        """
        Observe current market conditions and store observations.
        This method can be overridden by subclasses for more specific observations.
        """
        # Collect basic market data
        self.market_observations = {
            'price': self.model.get_current_price(),
            'trend': self.model.get_market_trend(),
            'volatility': self.model.get_market_volatility(),
            'step': self.model.steps,
            'macro_data': self.model.macro_data
        }

    def make_decisions(self):
        """
        Make decisions based on market observations.
        This method should be implemented by subclasses.
        """
        pass

    def buy_tokens(self, amount, price):
        """
        Place a buy order for tokens at the given price.

        Args:
            amount: Number of tokens to buy
            price: Price per token

        Returns:
            Order: The created order, or None if the order couldn't be placed
        """
        cost = amount * price
        if cost > self.dollars_held:
            return None

        # Reserve the dollars for this order
        self.dollars_held -= cost

        # Place the order in the order book
        order = self.model.place_order(self, 'buy', price, amount)

        return order

    def sell_tokens(self, amount, price):
        """
        Place a sell order for tokens at the given price.

        Args:
            amount: Number of tokens to sell
            price: Price per token

        Returns:
            Order: The created order, or None if the order couldn't be placed
        """
        if amount > self.tokens_held:
            return None

        # Reserve the tokens for this order
        self.tokens_held -= amount

        # Place the order in the order book
        order = self.model.place_order(self, 'sell', price, amount)

        return order

    def enter_market(self):
        """
        Enter the market.
        """
        self.active = True

    def exit_market(self):
        """
        Exit the market.
        """
        self.active = False

    def check_transactions(self):
        """
        Check if any transactions were completed in the previous step.

        Returns:
            dict: A dictionary with 'buys' and 'sells' lists containing completed transactions
        """
        current_step = self.model.steps -1

        # Get transactions from the current step
        current_transactions = {
            'buys': [],
            'sells': []
        }

        for transaction in self.transaction_history:
            if transaction['step'] == current_step:
                if transaction['type'] == 'buy':
                    current_transactions['buys'].append(transaction)
                elif transaction['type'] == 'sell':
                    current_transactions['sells'].append(transaction)

        return current_transactions


class HodlerAgent(BaseAgent):
    """
    Hodler agent class for blockchain token value simulation.
    Hodlers buy believing that the long-term trajectory is upward indefinitely.
    They buy and hold tokens indefinitely.
    """
    def __init__(self, model, tokens_held:int=0, dollars_held:float=0, sell_threshold:float=0.9):
        """
        Initialize a new HodlerAgent.

        Args:
            model: The model instance the agent belongs to
            tokens_held: Initial number of tokens held by the agent
            dollars_held: Initial amount of dollars held by the agent
            buy_threshold: Threshold for buying decision (0-1)
            sell_threshold: Threshold for selling decision (0-1)
        """
        super().__init__(model, tokens_held, dollars_held)
        self.sell_threshold = sell_threshold
        self.confidence = 0.5  # Initial confidence in the market (0-1)
        self.exit_trigger = 0

    def observe_market(self):
        """
        Observe current market conditions and update confidence.
        """
        # Call the parent method to populate market_observations
        super().observe_market()

        # Update confidence based on market conditions
        # This is a simple example - in a real model, this would be more complex
        market_trend = self.market_observations.get('trend', 0)
        self.confidence = (self.confidence * 0.8) + (market_trend * 0.2)
        if random.random() < 0.05:
            self.confidence = 0 # random.random() as a life reason to exit the market

        # Add confidence to market observations
        self.market_observations['confidence'] = self.confidence


    def make_decisions(self):
        """
        Make decisions based on market observations.
        """
        # Check if any transactions were completed in the current step
        transactions = self.check_transactions()

        # Get market data from market_observations
        price = self.market_observations.get('price', 0)

        # Calculate buy and sell prices
        buy_price = round(price * (1 + random.uniform(-0.1 if price >0.1 else -0.2, 0.1 if price >0.1 else 0.2)), 2)
        sell_price = round(price * random.uniform(0.9 if price >0.1 else 0.8, 0.99), 2)

        # Only exit market if all tokens are sold (either through order or previous transactions)
        if self.tokens_held < 1 and len(transactions['sells']) > 0:
            self.exit_market()
            return

        if self.exit_trigger:
            if self.tokens_held > 0:
                self.sell_tokens(self.tokens_held, sell_price)
            else:
                self.exit_market()
            return

        # If confidence is low enough, sell tokens (liquidate)
        if self.confidence < self.sell_threshold and self.tokens_held > 0:
            self.exit_trigger = 1
            # Sell all tokens (liquidate)
            if self.tokens_held > 1:
                order = self.sell_tokens(self.tokens_held, sell_price)
        elif self.dollars_held > buy_price:
            # Buy tokens with a portion of available dollars
            if self.dollars_held < 2 * buy_price:
                amount_to_spend = self.dollars_held
            else:
                amount_to_spend = self.dollars_held * 0.5
            if buy_price > 0:
                amount_to_buy = int(amount_to_spend / buy_price)
                self.buy_tokens(amount_to_buy, buy_price)

        if self.tokens_held < 1 and self.dollars_held < self.market_observations.get('price', 0):
            self.exit_market()

class SpeculatorAgent(BaseAgent):
    """
    Speculator agent class for blockchain token value simulation.
    Speculators hope to make money arbitraging the market.
    """
    def __init__(self, model, tokens_held:int=0, dollars_held:float=0, risk_tolerance:float=0.5):
        """
        Initialize a new SpeculatorAgent.

        Args:
            model: The model instance the agent belongs to
            tokens_held: Initial number of tokens held by the agent
            dollars_held: Initial amount of dollars held by the agent
            risk_tolerance: Risk tolerance of the agent (0-1)
        """
        super().__init__(model, tokens_held, dollars_held)
        self.risk_tolerance = risk_tolerance
        self.market_analysis = {
            'trend': 0,
            'volatility': 0,
            'opportunity': 0
        }
        self.exit_trigger = 0

    def observe_market(self):
        """
        Observe current market conditions and perform market analysis.
        """
        # Call the parent method to populate market_observations
        super().observe_market()

        # Calculate arbitrage opportunity
        self.market_observations['opportunity'] = self.calculate_opportunity()

    def calculate_opportunity(self):
        """
        Calculate arbitrage opportunity based on market conditions.

        Returns:
            float: Opportunity score (0-1)
        """
        # This is a simple example - in a real model, this would be more complex
        trend = self.market_observations['trend']
        volatility = self.market_observations['volatility']

        # Higher volatility and stronger trend means more opportunity
        opportunity = (abs(trend) * 0.5) + (volatility * 0.5)
        return min(1.0, opportunity)

    def make_decisions(self):
        """
        Make decisions based on market analysis.
        """
        # Check if any transactions were completed in the current step
        transactions = self.check_transactions()

        # Get market data from market_observations
        price = self.market_observations.get('price', 0)
        opportunity = self.market_observations.get('opportunity', 0)
        trend = self.market_observations.get('trend', 0)

        if self.tokens_held < 1 and len(transactions['sells']) > 0:
            self.exit_market()
            return

        if self.exit_trigger:
            if self.tokens_held > 0:
                self.sell_tokens(self.tokens_held, round(price * random.uniform(0.9 if price >0.1 else 0.8,1), 2))
            else:
                self.exit_market()
            return

        if random.random() < 0.05: # random sell all event
            if self.tokens_held > 0:
                self.sell_tokens(self.tokens_held, round(price * random.uniform(0.9 if price >0.1 else 0.8,1), 2))
            self.exit_trigger = 1
            return

        if len(transactions["buys"]) == 0:
            amount_to_spend = self.dollars_held * self.risk_tolerance
            price_to_buy = round(price * (1 + (trend * 0.1)), 2)
            amount_to_buy = amount_to_spend / price_to_buy
            self.buy_tokens(amount_to_buy, price_to_buy)

        # If opportunity is high enough and trend is positive, buy
        if opportunity > 0.7 and trend >= 0 and self.dollars_held > 0:
            # Buy tokens with a portion of available dollars based on risk tolerance
            amount_to_spend = self.dollars_held * self.risk_tolerance
            if price > 0:
                price_to_buy = round(price * (1 + (trend * 0.1)), 2)
                amount_to_buy = amount_to_spend / price_to_buy
                self.buy_tokens(amount_to_buy, price_to_buy)

        # If opportunity is high enough and trend is negative, sell
        elif opportunity > 0.7 and trend <= 0 and self.tokens_held > 0:
            # Sell tokens based on risk tolerance
            if self.market_observations.get('step',0) > 5:
                amount_to_sell = self.tokens_held * self.risk_tolerance
                price_to_sell = round(price * random.uniform(0.95 if price >0.1 else 0.8,1.05 if price >0.1 else 1.2), 2)
                self.sell_tokens(amount_to_sell, price_to_sell)

        # If opportunity is very low, exit market
        elif opportunity < 0.25 and self.tokens_held > 0:
            # Liquidate (sell all tokens and exit)
            self.exit_trigger = 1
            order = self.sell_tokens(self.tokens_held, round(price * random.uniform(0.9, 0.99), 2))

            # Only exit market if all tokens are sold (either through order or previous transactions)
            if self.tokens_held < 1 and len(transactions['sells']) > 0:
                self.exit_market()
                return

        if len(transactions['buys'])>0 and (price/transactions["buys"][0]["price"] < 1-self.risk_tolerance or price/transactions["buys"][0]["price"] > 1+self.risk_tolerance):
            self.exit_trigger = 1
            order = self.sell_tokens(self.tokens_held, round(price * random.uniform(0.9, 0.99), 2))

        if self.tokens_held < 1 and self.dollars_held < self.market_observations.get('price', 0):
            self.exit_market()

class ImmediateUserAgent(BaseAgent):
    """
    Immediate User agent class for blockchain token value simulation.
    Immediate Users want to use the token immediately after purchase.
    """
    def __init__(self, model, tokens_held:int=0, dollars_held:float=0, usage_rate:float=0.8, max_token_price:float=2.0):
        """
        Initialize a new ImmediateUserAgent.

        Args:
            model: The model instance the agent belongs to
            tokens_held: Initial number of tokens held by the agent
            dollars_held: Initial amount of dollars held by the agent
            usage_rate: Rate at which tokens are used (0-1)
        """
        super().__init__(model, tokens_held, dollars_held)
        self.buy_attempt = 0
        self.usage_rate = usage_rate
        self.usage_history = []
        self.max_token_price = max_token_price

    def observe_market(self):
        """
        Observe current market conditions.
        """
        # Call the parent method to populate market_observations
        super().observe_market()

        # Immediate users are less concerned with market conditions
        # They primarily need tokens for immediate use
        pass

    def make_decisions(self):
        """
        Make decisions based on token needs.
        """
        # Check if any transactions were completed in the current step
        transactions = self.check_transactions()
        current_step = self.market_observations.get('step', 0)
        # Get market data from market_observations
        price = self.market_observations.get('price', 0)
        self.max_token_price = np.round(self.max_token_price * (1 + (self.market_observations.get('cpi_increase', np.array([0.0])))),2).item()
        # If tokens are low, buy more
        if self.tokens_held < 1 and len(transactions['buys']) == 0:
            # Buy tokens with available dollars
            buy_price = round(min(price * random.uniform(1, 1.1 if price >0.1 else 1.2), self.max_token_price), 2)
            if buy_price == self.max_token_price:
                if random.random() < 0.2:
                    buy_price = self.max_token_price * random.uniform(1, 1.15)
            if buy_price > 0 and self.dollars_held > buy_price:
                amount_to_buy = int(self.dollars_held / buy_price)
                self.buy_tokens(amount_to_buy, buy_price)
                self.buy_attempt += 1
                if self.buy_attempt > 5 and len(transactions['buys']) == 0:
                    if self.tokens_held > 0:
                        self.sell_tokens(self.tokens_held, round(price * random.uniform(0.9 if price >0.1 else 0.8,1), 2))
                    else:
                        self.exit_market()
                    return

        # Use tokens only if we have them (either from before or from completed transactions)
        if self.tokens_held > 0:
            self.use_tokens()

        if self.tokens_held < 1 and len(transactions['buys']) > 0:
            self.exit_market()

    def use_tokens(self):
        """
        Use tokens for their intended purpose.
        """
        # Use a portion of tokens based on usage rate
        amount_to_use = int(min(self.tokens_held, self.tokens_held * self.usage_rate))
        self.tokens_held -= amount_to_use

        # Record usage
        self.usage_history.append({
            'amount': amount_to_use,
            'step': self.model.steps
        })
        self.usage_rate = 1

        # Send used tokens to environment agent if available
        self.model.environment_agent.receive_tokens(amount_to_use, self, "immediate_usage")

        # If all tokens are used, exit market
        if self.tokens_held < 1:
            self.exit_market()


class DelayedUserAgent(BaseAgent):
    """
    Delayed User agent class for blockchain token value simulation.
    Delayed Users want to use the token in the future and purchase now
    because they anticipate higher future costs.
    """
    def __init__(self, model, tokens_held:int=0, dollars_held:float=0,
                 future_usage_time:int=10, price_expectation:float=1.1, max_token_price:float=2.0,):
        """
        Initialize a new DelayedUserAgent.

        Args:
            model: The model instance the agent belongs to
            tokens_held: Initial number of tokens held by the agent
            dollars_held: Initial amount of dollars held by the agent
            future_usage_time: Number of steps in the future when tokens will be used
            price_expectation: Expected price increase factor
        """
        super().__init__(model, tokens_held, dollars_held)
        self.future_usage_time = future_usage_time
        self.price_expectation = price_expectation
        self.expected_price = 0
        self.purchase_time = None
        self.planned_usage_time = None
        self.usage_history = []
        self.buy_attempt = 0
        self.exit_trigger = 0
        self.max_token_price = max_token_price

    def observe_market(self):
        """
        Observe current market conditions and update price expectations.
        """
        # Call the parent method to populate market_observations
        super().observe_market()

        # Update price expectation based on market trend
        trend = self.market_observations.get('trend', 0)
        self.price_expectation = max(1.0, self.price_expectation * (1 + (trend * 0.1)))

    def make_decisions(self):
        """
        Make decisions based on future token needs and price expectations.
        """
        # Check if any transactions were completed in the current step
        transactions = self.check_transactions()

        # Get market data from market_observations
        current_step = self.market_observations.get('step', 0)
        price = self.market_observations.get('price', 0)
        self.max_token_price = np.round(self.max_token_price * (1 + (self.market_observations.get('cpi_increase', np.array([0.0])))),2).item()
        if self.tokens_held < 1 and len(transactions['sells']) > 0:
            self.exit_market()
            return

        if self.exit_trigger:
            if self.tokens_held > 0:
                self.sell_tokens(self.tokens_held, round(price * random.uniform(0.9 if price >0.1 else 0.8,1), 2))
            else:
                self.exit_market()
            return

        if self.tokens_held > 0:
            self.buy_attempt = 0

        expected_future_price = price * self.price_expectation

        # If no tokens held and no planned usage, consider buying for future use
        if self.tokens_held == 0 and self.planned_usage_time is None:

            self.buy_attempt += 1
            if self.buy_attempt > 3 and len(transactions['buys']) == 0:
                self.exit_market()
                return

            # If expected future price is higher and we have dollars, buy now

            if expected_future_price > price and self.dollars_held > price:
                amount_to_buy = int(self.dollars_held / price)
                order = self.buy_tokens(amount_to_buy, min(self.max_token_price, price))

                # If order was placed or transactions were completed, set planned usage time
                if order is not None or len(transactions['buys']) > 0:
                    self.purchase_time = current_step
                    self.planned_usage_time = current_step + self.future_usage_time

        # If it's time to use the tokens and we have tokens (either from before or from completed transactions)
        elif self.planned_usage_time is not None and current_step >= self.planned_usage_time and self.tokens_held > 0:
            self.use_tokens()
        elif self.tokens_held > 0 and expected_future_price <= price:
            amount_to_sell = self.tokens_held
            self.sell_tokens(amount_to_sell, self.price_expectation)
            self.exit_trigger = 1
        elif self.planned_usage_time is not None and current_step >= self.planned_usage_time and (self.tokens_held < 1):
            self.exit_market()
            return

        if self.tokens_held < 1 and len(self.check_transactions()['buys']) > 0:
            self.exit_market()

    def use_tokens(self):
        """
        Use tokens for their intended purpose.
        """
        if self.tokens_held > 0:
            # Use all tokens
            amount_used = self.tokens_held
            self.tokens_held = 0

            # Record usage
            self.usage_history.append({
                'amount': amount_used,
                'step': self.model.steps
            })

            # Send used tokens to environment agent if available
            if hasattr(self.model, 'environment_agent') and self.model.environment_agent is not None:
                # Environment agent receives tokens from this user
                self.model.environment_agent.receive_tokens(amount_used, self, "delayed_usage")

            # Reset planned usage
            self.planned_usage_time = None

            # If all tokens are used and no more dollars, exit market
            if self.tokens_held < 1:
                self.exit_market()


class EnvironmentAgent(BaseAgent):
    """
    Environment agent class for blockchain token value simulation.
    Acts as an avatar of the environment, performing various system-level actions:
    - Receives and burns tokens that users spend
    - Pays staking rewards to node operators
    - Sells tokens when certain events occur
    - Performs observation and calculation actions
    """
    def __init__(self, model, tokens_held:int=1000000, dollars_held:float=1000000, max_tokens:int=100000,
                 token_supply_per_tick:list=None):
        """
        Initialize a new EnvironmentAgent.

        Args:
            model: The model instance the agent belongs to
            tokens_held: Initial number of tokens held by the agent (large supply)
            dollars_held: Initial amount of dollars held by the agent (large supply)
        """
        super().__init__(model, tokens_held, dollars_held)
        #self.burned_tokens = 0  # Track burned tokens
        #self.granted_tokens = 0  # Track granted tokens
        if token_supply_per_tick is None:
            token_supply_per_tick = []
        self.received_tokens = 0  # Track received tokens
        self.sold_tokens = 0  # Track sold tokens
        self.token_events = []  # Track token-related events
        self.market_observations = {}  # Store market observations
        self.minted_tokens = 0
        self.max_tokens = max_tokens
        self.token_supply_per_tick = token_supply_per_tick

    def mint_tokens(self, amount_needed:int, reason:str):
        if self.minted_tokens + amount_needed > self.max_tokens:
            amount_needed = self.max_tokens - self.minted_tokens
        if amount_needed > 0:
            self.tokens_held += amount_needed
            self.minted_tokens += amount_needed
            current_step = self.market_observations.get('step', 0)
            self.token_events.append({
                'type': 'mint',
                'amount': amount_needed,
                'reason': reason,
                'step': current_step
            })

    def observe_market(self):
        """
        Observe current market conditions and store observations.
        """
        # Call the parent method to populate market_observations
        super().observe_market()


        # Record observation event
        self.token_events.append({
            'type': 'observation',
            'data': self.market_observations,
            'step': self.model.steps
        })

    def make_decisions(self):
        """
        Make decisions based on market observations.
        This includes selling tokens when certain conditions are met
        and paying staking rewards to node operators.
        """
        # Check if any transactions were completed in the current step
        transactions = self.check_transactions()

        # Perform calculations based on market observations
        self.perform_calculations()

        # Sell tokens if market conditions warrant it
        self.sell_tokens_on_conditions()

    def perform_calculations(self):
        """
        Perform calculations based on market observations.
        """
        # Example calculation: predict future price based on trend
        if 'trend' in self.market_observations and 'price' in self.market_observations:
            trend = self.market_observations['trend']
            current_price = self.market_observations['price']
            predicted_price = current_price * (1 + trend * 0.1)

            # Record calculation event
            self.token_events.append({
                'type': 'calculation',
                'data': {
                    'predicted_price': predicted_price,
                    'current_price': current_price,
                    'trend': trend
                },
                'step': self.market_observations.get('step', 0)
            })

    def receive_tokens(self, amount, from_agent=None, reason="usage"):
        """
        Receive tokens from an agent (e.g., when users spend tokens).

        Args:
            amount: Number of tokens to receive
            from_agent: The agent sending the tokens (optional)
            reason: Reason for receiving tokens (default: "usage")

        Returns:
            bool: True if tokens were received successfully
        """
        if amount <= 0:
            return False

        # Increase token balance
        self.tokens_held += amount
        self.received_tokens += amount

        # Record token receipt event
        self.token_events.append({
            'type': 'receive',
            'amount': amount,
            'from_agent': from_agent.unique_id if from_agent else None,
            'reason': reason,
            'step': self.market_observations.get('step', 0)
        })

        return True

    # def burn_tokens(self, amount, reason="deflation"):
    #     """
    #     Burn tokens (remove them from circulation).
    #
    #     Args:
    #         amount: Number of tokens to burn
    #         reason: Reason for burning tokens (default: "deflation")
    #
    #     Returns:
    #         bool: True if tokens were burned successfully
    #     """
    #     if amount <= 0 or amount > self.tokens_held:
    #         return False
    #
    #     # Decrease token balance
    #     self.tokens_held -= amount
    #     self.burned_tokens += amount
    #
    #     # Record token burning event
    #     self.token_events.append({
    #         'type': 'burn',
    #         'amount': amount,
    #         'reason': reason,
    #         'step': self.model.steps
    #     })
    #
    #     return True

    # def grant_tokens(self, amount, to_agent, reason="staking_reward"):
    #     """
    #     Grant tokens to an agent (e.g., staking rewards).
    #
    #     Args:
    #         amount: Number of tokens to grant
    #         to_agent: The agent receiving the tokens
    #         reason: Reason for granting tokens (default: "staking_reward")
    #
    #     Returns:
    #         bool: True if tokens were granted successfully
    #     """
    #     if amount <= 0 or amount > self.tokens_held:
    #         return False
    #
    #     # Decrease token balance
    #     self.tokens_held -= amount
    #     self.granted_tokens += amount
    #
    #     # Increase recipient's token balance
    #     to_agent.tokens_held += amount
    #
    #     # Record token granting event
    #     self.token_events.append({
    #         'type': 'grant',
    #         'amount': amount,
    #         'to_agent': to_agent.unique_id,
    #         'reason': reason,
    #         'step': self.model.steps
    #     })
    #
    #     # Record transaction in recipient's history
    #     to_agent.transaction_history.append({
    #         'type': 'receive',
    #         'amount': amount,
    #         'from_agent': self.unique_id,
    #         'reason': reason,
    #         'step': self.model.steps
    #     })
    #
    #     return True

    def sell_tokens_on_conditions(self):
        """
        Sell tokens when certain market conditions are met.
        """
        # # Example conditions for selling tokens:
        # # 1. Market is trending upward strongly
        # # 2. Price is above a certain threshold
        # # 3. Random event (with low probability)
        #
        # trend = self.market_observations.get('trend', 0)
        price = self.market_observations.get('price', 0)
        #
        # # Strong upward trend and high price
        # if trend > 0.5 and price > self.model.initial_price * 1.5:
        #     amount_to_sell = min(self.tokens_held * 0.05, 1000)  # Sell up to 5% of holdings or 100 tokens
        #     if amount_to_sell > 0:
        #         self.sell_tokens(amount_to_sell, price)
        #         self.sold_tokens += amount_to_sell
        #
        #         # Record token selling event
        #         self.token_events.append({
        #             'type': 'market_sell',
        #             'amount': amount_to_sell,
        #             'price': price,
        #             'reason': "high_price_trend",
        #             'step': self.market_observations.get('step', 0)
        #         })
        # # Sell because received token number has exceeded 1 000 000
        #
        # # Random selling event (low probability)
        # elif random.random() < 0.02:  # 2% chance per step
        #     amount_to_sell = min(self.tokens_held * 0.02, 500)  # Sell up to 2% of holdings or 50 tokens
        #     if amount_to_sell > 0:
        #         self.sell_tokens(amount_to_sell, round(price*random.uniform(0.95,1.0),2))
        #         self.sold_tokens += amount_to_sell
        #
        #         # Record token selling event
        #         self.token_events.append({
        #             'type': 'market_sell',
        #             'amount': amount_to_sell,
        #             'price': price,
        #             'reason': "random_event",
        #             'step': self.market_observations.get('step', 0)
        #         })

        # sell received tokens
        if self.received_tokens > 0:
            selling_price = round(price*random.uniform(0.95,1.02),2)
            self.sell_tokens(self.received_tokens, selling_price)
            self.token_events.append({
                'type': 'market_sell',
                'amount': self.received_tokens,
                'price': selling_price,
                "reason": "received tokens sell attempt",
                "step": self.market_observations.get('step', 0),
            })
        self.sold_tokens += self.received_tokens

        # every month token supply
        if self.token_supply_per_tick[self.market_observations.get('step', 0)] > 0:
            self.mint_tokens(int(self.token_supply_per_tick[self.market_observations.get('step', 0)]), "token_supply_per_tick")
        selling_price = round(price * random.uniform(0.9,1.0), 2)
        self.sell_tokens(self.tokens_held, selling_price)
        self.sold_tokens += self.tokens_held

        # Record token selling event
        self.token_events.append({
            'type': 'market_sell',
            'amount': self.tokens_held,
            'price': price,
            'reason': "token_supply_per_tick sell attempt",
            'step': self.market_observations.get('step', 0)
        })
