from stable_baselines3 import DQN
from training.rl_models.ppo.train import LoggingCallback
import os

def train_dqn(env, db_session, total_timesteps=1000):
    # DQN is for discrete action spaces
    model = DQN("MlpPolicy", env, verbose=1)
    callback = LoggingCallback(db_session, "DQN")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'dqn_model.zip')
    model.save(save_path)
    return model


