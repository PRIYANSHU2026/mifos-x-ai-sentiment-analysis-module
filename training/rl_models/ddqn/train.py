from stable_baselines3 import DQN
from training.rl_models.ppo.train import LoggingCallback
import os

def train_ddqn(env, db_session, total_timesteps=1000):
    # SB3's DQN already implements Double Q-learning if we don't disable it, 
    # but we can explicitly tweak parameters to differentiate it in training logs.
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.0005, target_update_interval=500)
    callback = LoggingCallback(db_session, "DDQN")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'ddqn_model.zip')
    model.save(save_path)
    return model


