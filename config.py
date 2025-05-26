# Paths
IMAGE_DIR = "dataset_last/images"
MASKED_DIR = "dataset_last/masked"
MASK_DIR = "dataset_last/masks"
CHECKPOINT_DIR = "checkpoints_last"
OUTPUT_DIR = "outputs_last"

# Training hyperparameters
IMG_SIZE = 256
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 2e-4

LAMBDA_PERCEPTUAL = 0.05
LAMBDA_TV = 0.1

DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

# Logging
SAVE_SAMPLE_EVERY = 5