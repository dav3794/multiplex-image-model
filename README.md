# Multiplex image model

## Installation

Simply install the packages in provided requirements file:
```
pip install -r requirements.txt
```

## Data preparation

To use the model, you need to have your data in the following directories structure:

```
.
└── data
    ├── test
    │   ├── dataset1
    │   │   └── imgs
    │   ├── dataset2
    │   │   └── imgs
    │   └── dataset3
    │       └── imgs
    └── train
        ├── dataset1
        │   └── imgs
        ├── dataset2
        │   └── imgs
        └── dataset3
            └── imgs

```
Where `train` and `test` are splits and `dataset...` is a dataset/panel with images. Provide full paths to your splits directories in `congifs/all_panels_config.yaml` under `paths`, as in provided example config. Then list all the datasets subdirectories you want to use from your split paths under `datasets` config key. Ultimately, for each dataset used, provide a list of markers that are represented in consequtive image channels for that dataset, in the `markers` config section. 

If you are working on rudy, use the provided `split_data.ipynb` notebook to do the dataset split (only the section `Images preparation`).

## Training a model

Training utilizes the Neptune logging. You can make free account with academic license (https://neptune.ai/research), then insert your project and api_token into `train_masked_config.yaml` under Neptune configuration.


To train a model, simply execute training script with its config file:
```
python3 train_masked_model.py train_masked_config.yaml
```


