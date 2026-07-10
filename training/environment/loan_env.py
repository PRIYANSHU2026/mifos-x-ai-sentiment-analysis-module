import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class LoanEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render_modes': ['human']}

    def __init__(self, data_path, continuous_action=False):
        super(LoanEnv, self).__init__()
        # Load dataset
        self.df = pd.read_csv(data_path)
        # Drop target if present
        if 'Loan_Status' in self.df.columns:
            self.features = self.df.drop(columns=['Loan_Status']).values
            self.labels = self.df['Loan_Status'].values
        else:
            self.features = self.df.values
            self.labels = np.ones(len(self.df)) # Dummy
            
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        
        self.continuous_action = continuous_action
        
        # Action space
        if self.continuous_action:
            # Action: [interest_rate] from 0 to 1 (mapped to 5% to 25%, < 0.1 means reject)
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            # 0: Reject, 1: Approve 8%, 2: Approve 12%, 3: Approve 16%, 4: Approve 20%
            self.action_space = spaces.Discrete(5)
            
        # State space (10 features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.features.shape[1],), dtype=np.float32
        )

    def step(self, action):
        state = self.features[self.current_step]
        actual_repayment = self.labels[self.current_step] # 1 if repaid (approved in original), 0 if defaulted/rejected
        
        # Default assumptions
        loan_amount = state[5] # assuming standard order
        
        # We need a risk proxy. Repayment_History (index 4) is a good proxy.
        # But we'll just use the actual_repayment from the dataset to simulate the outcome.
        
        reward = 0
        interest_rate = 0.0
        approved = False
        
        if self.continuous_action:
            val = action[0]
            if val < 0.1:
                approved = False
            else:
                approved = True
                interest_rate = 0.05 + ((val - 0.1) / 0.9) * 0.20 # 5% to 25%
        else:
            if action == 0:
                approved = False
            elif action == 1:
                approved = True
                interest_rate = 0.08
            elif action == 2:
                approved = True
                interest_rate = 0.12
            elif action == 3:
                approved = True
                interest_rate = 0.16
            elif action == 4:
                approved = True
                interest_rate = 0.20
                
        # Reward calculation
        if not approved:
            # Rejecting
            if actual_repayment == 1:
                # Rejected a good customer
                reward = -0.5
            else:
                # Correctly rejected a bad customer
                reward = 0.5
        else:
            # Approving
            if actual_repayment == 1:
                # Good customer, profit = interest
                # Higher interest = higher reward, but maybe penalize very high rates for fairness
                profit = interest_rate * 10 # Scaled profit
                fairness_penalty = max(0, (interest_rate - 0.15) * 5)
                reward = profit - fairness_penalty
            else:
                # Defaulted
                # Loss = principal
                loss_penalty = -5.0
                reward = loss_penalty
                
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Info dictionary
        info = {
            'approved': approved,
            'interest_rate': interest_rate,
            'actual_repayment': actual_repayment,
            'reward': reward
        }
        
        next_state = self.features[self.current_step] if not terminated else np.zeros(self.observation_space.shape)
        
        return next_state.astype(np.float32), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        state = self.features[self.current_step]
        return state.astype(np.float32), {}

    def render(self):
        pass
