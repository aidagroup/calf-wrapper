uv run run/eval.py cartpole \
--eval_mode calf_wrapper \
--model_path run/artifacts/ppo_CartpoleSwingupEnv-v0_42/checkpoints/ppo_checkpoint_99000_steps.zip \
--calf.relaxprob_init 0.0 \
--mlflow.experiment_name eval_cartpole \
--mlflow.run_name calf_wrapper_conservative_early
