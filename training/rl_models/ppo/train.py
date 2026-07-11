from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os

class LoggingCallback(BaseCallback):
    def __init__(self, db_session, model_name, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.db_session = db_session
        self.model_name = model_name
        self.episode = 0

    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None and self.locals["dones"][0]:
            self.episode += 1
            reward = self.locals["rewards"][0]
            # Simple logging, real loss needs to be extracted from logger if available
            loss = 0.0 # Placeholder
            learning_rate = self.model.lr_schedule(self.model.num_timesteps)
            
            # Using raw SQL insertion to avoid circular imports here, or import the model
            # For simplicity, we just use the session.
            from database.models import TrainingHistory
            history = TrainingHistory(
                model_name=self.model_name,
                episode=self.episode,
                reward=float(reward),
                loss=float(loss),
                learning_rate=float(learning_rate)
            )
            self.db_session.add(history)
            self.db_session.commit()
        return True

def train_ppo(env, db_session, total_timesteps=1000):
    model = PPO("MlpPolicy", env, verbose=1)
    callback = LoggingCallback(db_session, "PPO")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'ppo_model.zip')
    model.save(save_path)
    return model


