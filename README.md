# Copepod segmentation inference

## Model download

From [GitHub release page](https://github.com/Appsilon/copepods-lipid-content/releases/) download the model files.
Put them into `models` subfolder so this directory looks more or less like:

```
.
├── example_inputs
│   │...
│   └── 20130902 221000 109 000001 0299 0118.bmp
├── example_segmentations
│   │...
│   └── 20130902 221000 109 000001 0299 0118.bmp
├── inference.ipynb
├── models
│   ├── learner.pkl
│   └── model_resnet34.pth
├── README.md
└── requirements.txt
```

## Inference notebook

To run this notebook you have to create the following environment:

```
conda create -n loki-inference python=3.9
conda activate loki-inference
pip install -r requirements.txt
```

Then, use the [inference](inference.ipynb) notebook.
