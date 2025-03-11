# 1. Import required modules
from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import Draem

# 2. Create a dataset
# MVTec is a popular dataset for anomaly detection
datamodule = MVTec(
    root="./datasets/MVTec",  # Path to download/store the dataset
    category="leather",  # MVTec category to use
    train_batch_size=4,  # Number of images per training batch
    eval_batch_size=4,  # Number of images per validation/test batch
    num_workers=4,  # Number of parallel processes for data loading
)

# 3. Initialize the model
# EfficientAd is a good default choice for beginners
model = Draem()

# 4. Create the training engine
engine = Engine(max_epochs=100)  # Train for 10 epochs

# 5. Train the model
engine.fit(datamodule=datamodule, model=model)