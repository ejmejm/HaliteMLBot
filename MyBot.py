import reinforcementLearner
from sys import argv

# Load the model from the models directory. Models directory is created during training.
# Run "make" to download data and train.
if len(argv) <= 1 or argv[1] == "None":
	loc = None
else:
	loc = argv[1]
reinforcementLearner.Bot(location=loc, name="Ejmejm").play()
