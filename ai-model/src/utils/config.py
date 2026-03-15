# Central configuration
DATA_ROOT = "/content/drive/MyDrive/GP/Code/Seminar2/datasets/scvd_dataset/SCVD/"

NUM_CLASSES = 3
SEQ_LENGTH = 30
IMAGE_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-4

CLASS_NAMES = ["Normal", "Violence", "Weaponized"]


#The system was implemented using a modular source code structure
#separating dataset handling, model definition, training utilities, evaluation metrics,
# and object detection logic to ensure scalability and maintainability.
