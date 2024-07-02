# Project Report

## Link to Kaggle Notebook

https://www.kaggle.com/code/furqanshuja/individual-project

## Project Overview

This project focuses on developing a modular object detection model using the CISOL dataset. The goal is to fit the best possible model by experimenting with different hyperparameters and architectures, and to submit the most promising results to the EvalAI leaderboard.

## Repository Structure

```python
DLCV24-INDIVIDUAL-PROJECT
├── src
│   └── dlcv
│       ├── __pycache__
│       ├── __init__.py
│       ├── config.py        # Holds all configuration parameters
│       ├── dataset1.py      # Processes train data
│       ├── dataset2.py      # Processes test data
│       ├── inference.py     # Evaluates and builds JSON file for prediction on set data
│       ├── model.py         # Defines all models and their backbones
│       ├── plot.py          # Uses the saved training loss to make a plot
│       ├── train.py         # Uses configurable parameters and calls relevant functions for training the model
│       ├── training.py      # Contains the functionality for training with set epochs and saves training loss
│       ├── utils.py         # Holds all side functionalities necessary for training and evaluation
│       ├── visualize.py     # Uses the trained model to predict and visualize bounding boxes on test data
│   ├── dlcv.egg-info
│   ├── __init__.py
├── .gitignore
├── individual-project.ipynb
├── pyproject.toml
├── README.md
└── setup.py
```

## Installation on Kaggle

The repository was installed on Kaggle using following script

```python
token = "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" 
user = "BUW-CV"
repo_name = "dlcv24-individual-final-project-FurqanShuja"
url = f"https://{user}:{token}@github.com/{user}/{repo_name}.git"
!git clone {url}
!pip install /kaggle/working/dlcv24-individual-final-project-FurqanShuja
```

## Usage Instructions

Script to run train.py

```python
import dlcv.train as train
import dlcv.config as config
cfg = config.get_cfg_defaults()

# Basic Customization
cfg.TRAIN.BASE_LR = 0.004
cfg.TRAIN.WEIGHT_DECAY = 0.005
cfg.TRAIN.EPOCHS = 20
cfg.TRAIN.BATCH_SIZE = 2

# Model Selection
cfg.MODEL.TYPE = "fasterrcnn"

# Warmup options
cfg.TRAIN.WARMUP_LR = 0.0004
cfg.TRAIN.WARMUP_EPOCHS = 3

# Scheduling Options
cfg.TRAIN.PCT_START = 0.3
cfg.TRAIN.ANNEAL_STRATEGY = 'cos'

# Augmentation Techniques Values
cfg.TRAIN.HORIZONTAL_FLIP_PROB = 0.1
cfg.TRAIN.ROTATION_DEGREES = 3

train.main(cfg)
```

Script to run inference.py

```python
import dlcv.inference as inf
import dlcv.config as config
cfg = config.get_cfg_defaults()

# Basic Customization
cfg.TEST.SCORE_THRESHOLD = 0
cfg.TEST.IOU_THRESHOLD = 1
cfg.TEST.BATCH_SIZE = 10

cfg.MODEL.TYPE = "fasterrcnn"
inf.main(cfg)
```

Script to run visualize.py

```python
import dlcv.visualize as vis
import dlcv.config as config
cfg = config.get_cfg_defaults()

# Basic Customization
cfg.TEST.SCORE_THRESHOLD = 0
cfg.TEST.IOU_THRESHOLD = 1
cfg.TEST.BATCH_SIZE = 10
cfg.MODEL.TYPE = "fasterrcnn"

vis.main(cfg)
```

## Modular Design 

These are included inside config.py file
```python
 _C.MODEL = CN()
    _C.MODEL.NUM_CLASSES = 6
    _C.MODEL.TYPE = "fasterrcnn"  # 3 models available fasterrcnn, fcos and retinanet, with backbone resnet50
```

## Hyperparameter Configuration

Inside the config.py, the important hyperparameters are:
```python
# Major Hyperparameters for fine-tuning train
    _C.TRAIN = CN()
    _C.TRAIN.BATCH_SIZE = 4
    _C.TRAIN.BASE_LR = 0.004
    _C.TRAIN.WEIGHT_DECAY = 0.005
    _C.TRAIN.NUM_EPOCHS = 20

    # Warmup Options
    _C.TRAIN.WARMUP_LR = 0.0004
    _C.TRAIN.WARMUP_EPOCHS = 1

    # Scheduling Options
    _C.TRAIN.PCT_START = 0.3
    _C.TRAIN.ANNEAL_STRATEGY = 'cos'

    # Train Data Augmentation Techniques
    _C.TRAIN.IMAGE_HEIGHT = 3509
    _C.TRAIN.IMAGE_WIDTH = 2480
    _C.TRAIN.HORIZONTAL_FLIP_PROB = 0
    _C.TRAIN.ROTATION_DEGREES = 0

    # Customizing evaluation on test data
    _C.TEST = CN()
    _C.TEST.SCORE_THRESHOLD=0
    _C.TEST.IOU_THRESHOLD=1
    _C.TEST.IMAGE_HEIGHT = 3509
    _C.TEST.IMAGE_WIDTH = 2480
```

## Experimentation
### Experimental Design

Multiple experiments were conducted by varying hyperparameters and model architectures. The results were evaluated based on metrics like mAP (mean Average Precision). 

Versions on kaggle have been named for understanding. Furthermore the details for each run is as follows. 

### Key Experiments and Results

| Parameter               | Experiment 1 | Experiment 2 | Experiment 3 | Experiment 4 | Experiment 5 | Experiment 6 | Experiment 7 | Best Run (Experiment 3) |
|-------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------------------|
| Kaggle Version          | Version 3    | Version 9    | Version 5    | Version 7    | Version 4    | Version 6    | Version 8    | Version 5                |
| Model                   | RetinaNet    | RetinaNet    | FasterRCNN   | FasterRCNN   | FCOS         | FCOS         | FCOS         | FasterRCNN               |
| Backbone                | ResNet50     | ResNet50     | ResNet50     | ResNet50     | ResNet50     | ResNet50     | ResNet50     | ResNet50                 |
| Learning Rate           | 0.004        | 0.004        | 0.004        | 0.004        | 0.004        | 0.004        | 0.004        | 0.004                    |
| Weight Decay            | 0.005        | 0.005        | 0.005        | 0.005        | 0.005        | 0.005        | 0.005        | 0.005                    |
| Epochs                  | 30           | 20           | 50           | 20           | 30           | 100          | 20           | 50                       |
| Batch Size              | 2            | 2            | 2            | 2            | 2            | 2            | 2            | 2                        |
| Warmup                  | None         | LR: 0.0004, Epochs: 3 | None  | LR: 0.0004, Epochs: 3 | None | None        | LR: 0.0004, Epochs: 3 | None                    |
| Augmentations           | None         | Horizontal Flip, Rotation | None | Horizontal Flip, Rotation | None | None | Horizontal Flip, Rotation | None                    |
| Other parameters        | Default Config.py | Default Config.py | Default Config.py | Default Config.py | Default Config.py | Default Config.py | Default Config.py | Default Config.py        |
| Results (mAP)           | 49.6         | 48.3         | 56.95        | 51.74        | 23.9         | 25.73        | 19.7         | 56.95                    |



#### Train-loss Plot Combined

Train Losses for each run were plotted on kaggle in respective versions. For a short comparison, the train losses for best runs for each model have been combined in the following plot.

![alt text](images/TRAIN_LOSS_COMBINED_PLOT.png)

#### Visualisation on Test Data

Visualization for each run was done on kaggle in respective versions. For a short comparison, one visualization per run have been added in the following images.

![alt text](images/2.png)
![alt text](images/3.png)
![alt text](images/4.png)

## EvalAI Leaderboard Submissions

All the predictions made by experiment runs discussed before were submitted to EVALAI. The mAP Score from various models can be seen in the following Visualisation. 

![alt text](images/1.png)

## Conclusion

The project successfully developed a modular and customizable object detection framework. Through extensive experimentation, it was concluded that FasterRCNN with resnet-50 gave the best score. 
