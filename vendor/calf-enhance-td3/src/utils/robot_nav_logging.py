import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.utils.artifact_uploader import get_artifact_uploader


def _extract_robot_nav_arrays(episode_trajectory):
    if not episode_trajectory:
        return None, None, None

    obs_list = []
    for step in episode_trajectory:
        obs = np.asarray(step["obs"])
        if obs.ndim == 2:
            obs = obs[0]
        obs_list.append(obs)

    if not obs_list:
        return None, None, None

    obs_array = np.asarray(obs_list)
    positions = obs_array[:, 0:2]
    goal = obs_array[0, 4:6]

    obstacle_flat = obs_array[0, 6:]
    if obstacle_flat.size == 0:
        obstacles = np.zeros((0, 3))
    else:
        obstacles = obstacle_flat.reshape(-1, 3)
        obstacles = obstacles[obstacles[:, 2] > 0]

    return positions, goal, obstacles


def _extract_total_reward(episode_trajectory):
    total = 0.0
    for step in episode_trajectory:
        reward = step.get("reward", 0.0)
        reward_arr = np.asarray(reward)
        if reward_arr.size == 0:
            continue
        total += float(reward_arr.reshape(-1)[0])
    return total


def log_robot_nav_trajectory(
    episode_trajectory,
    global_step,
    total_reward=None,
    goal_reached=False,
    artifact_subdir="robot_navigation",
):
    positions, goal, obstacles = _extract_robot_nav_arrays(episode_trajectory)
    if positions is None:
        return

    if total_reward is None:
        total_reward = _extract_total_reward(episode_trajectory)

    status = "Goal Reached" if goal_reached else "Goal Missed"
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_title(
        f"Robot Navigation Trajectory @ step {global_step}\nReturn={total_reward:.2f} | {status}"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.plot(
        positions[:, 0], positions[:, 1], color="#1f77b4", linewidth=2, label="Path"
    )
    ax.scatter(
        positions[0, 0],
        positions[0, 1],
        color="#2ca02c",
        marker="o",
        s=50,
        label="Start",
    )
    ax.scatter(goal[0], goal[1], color="#d62728", marker="*", s=120, label="Goal")

    for x, y, radius in obstacles:
        circle = plt.Circle(
            (x, y), radius, color="#ff9896", ec="#c44e52", alpha=0.5, linewidth=1.5
        )
        ax.add_patch(circle)

    ax.grid(True, alpha=0.3)

    uploader = get_artifact_uploader()
    if uploader is None:
        raise RuntimeError(
            "ArtifactUploader not initialized. Call init_artifact_uploader first."
        )

    with uploader.lock:
        target_dir = uploader.staging_dir / artifact_subdir / "trajectories"
        target_dir.mkdir(parents=True, exist_ok=True)
        plot_path = target_dir / f"robot_nav_traj_{global_step:010d}.png"
        fig.savefig(plot_path, bbox_inches="tight")

    plt.close(fig)
