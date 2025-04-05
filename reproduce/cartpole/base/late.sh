uv run run/eval.py cartpole \
--eval_mode base \
--model_path run/artifacts/ppo_CartpoleSwingupEnv-v0_42/checkpoints/ppo_checkpoint_270000_steps.zip \
--mlflow.experiment_name eval_cartpole \
--mlflow.run_name base_late