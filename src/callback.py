import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Use pandas to save data

from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import mlflow


class PlottingCallback(BaseCallback):
    def __init__(
        self,
        save_path: Path,
        update_every_episodes: int = 1,
        verbose: int = 0,
        is_console_mode: bool = True,
    ):
        super(PlottingCallback, self).__init__(verbose)
        self.episode_rewards = []  # List to store rewards for each episode
        self.episodes = []  # List to store episode numbers

        self.update_every_episodes = update_every_episodes

        self.current_episode_reward = 0  # Accumulator for the current episode reward
        self.save_path = save_path  # Path to save the reward data
        self.is_console_mode = is_console_mode

        if not is_console_mode:
            # Set up the plot
            self.figure, self.ax = plt.subplots()
            (self.line,) = self.ax.plot([], [], label="Episode Reward")
            self.ax.set_xlabel("Episodes")
            self.ax.set_ylabel("Rewards")
            self.ax.set_title("Live Learning Curve")
            self.ax.legend()
            plt.show(block=False)

    def _on_step(self) -> bool:
        # Accumulate the reward from the current step
        rewards = self.locals["rewards"]
        self.current_episode_reward += np.sum(
            rewards
        )  # Sum rewards from all environments

        # Check if any episode has ended
        dones = self.locals["dones"]
        infos = self.locals["infos"]  # This is now a list of dictionaries

        if np.any(dones):
            # Check if any episode was truncated or ended due to a time limit
            if np.any([info.get("TimeLimit.truncated", False) for info in infos]):
                # Log the total reward for the episode
                self.episode_rewards.append(self.current_episode_reward)
                self.episodes.append(len(self.episodes) + 1)
                mlflow.log_metric(
                    "detail/episode_reward",
                    self.current_episode_reward,
                    step=len(self.episodes),
                )
                # Record the episode reward to the logger
                self.logger.record("train/episode_reward", self.current_episode_reward)

                self.current_episode_reward = 0  # Reset for the next episode

                # Only update the plot every `update_every_episodes` episodes
                if (
                    not self.is_console_mode
                    and len(self.episodes) % self.update_every_episodes == 0
                ):
                    self.line.set_xdata(self.episodes)
                    self.line.set_ydata(self.episode_rewards)
                    self.ax.relim()
                    self.ax.autoscale_view()
                    plt.draw()
                    plt.pause(0.01)

        return True

    def _on_training_end(self) -> None:
        # Save the episode rewards to a CSV file
        df = pd.DataFrame({"Episode": self.episodes, "Reward": self.episode_rewards})
        df.to_csv(self.save_path, index=False)
