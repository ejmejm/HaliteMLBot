import reinforcementLearner

# Load the model from the models directory. Models directory is created during training.
# Run "make" to download data and train.
reinforcementLearner.Bot(location="rl_training.ckpt", name="EjmejmCPU", processor="CPU").play()
