uv run run/eval.py pendulum \
--eval_mode base \
--model_path run/artifacts/ppo_Pendulum-v1_9/checkpoints/ppo_checkpoint_36000_steps.zip \
--mlflow.experiment_name eval_pendulum \
--mlflow.run_name base_mid
