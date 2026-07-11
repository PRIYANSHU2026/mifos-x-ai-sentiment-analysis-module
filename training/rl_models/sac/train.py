from stable_baselines3 import SAC
from training.rl_models.ppo.train import LoggingCallback
import os

def train_sac(env, db_session, total_timesteps=1000):
    # SAC is for continuous action spaces
    model = SAC("MlpPolicy", env, verbose=1)
    callback = LoggingCallback(db_session, "SAC")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'sac_model.zip')
    model.save(save_path)
    return model


