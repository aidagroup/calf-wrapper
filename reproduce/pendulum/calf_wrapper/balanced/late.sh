uv run run/eval.py pendulum \
--eval_mode calf_wrapper \
--model_path run/artifacts/ppo_Pendulum-v1_9/checkpoints/ppo_checkpoint_102000_steps.zip \
--calf.relaxprob_init 0.5 \
--mlflow.experiment_name eval_pendulum \
--mlflow.run_name calf_wrapper_balanced_late
