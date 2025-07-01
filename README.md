# Bio-Inspired-Computer-Vision (Still in Progress)

This repository contains code for training and evaluating a spiking neural network (SNN) with a spiking RNN region‑of‑interest (ROI) predictor, alongside a CNN baseline, on voxelised event‑based vision data.

## Dissertation project

This work forms part of a dissertation at the University of Manchester. It explores how SNNs can leverage ROI prediction to improve classification on event‑based vision datasets.

## Features

* Spiking RNN ROI predictor integrated with a LeNet‑style SNN
* CNN baseline for comparison
* Real‑time progress bars with `tqdm` for clear training and evaluation feedback
* Class‑weighted loss to balance skewed class frequencies
* Evaluation scripts producing confusion matrices and classification reports

## Repository structure

```
.
├── data               # Dataset loaders and transforms
│   ├── datasets.py
│   └── transforms.py
├── models             # Network definitions
│   ├── spiking_rnn_roi.py
│   ├── lenet_snn.py
│   └── cnn_baseline.py
├── scripts            # Training and evaluation scripts
│   ├── train_snn.py
│   ├── train_cnn.py
│   └── evaluate.py
├── results            # Saved models and output figures
│   ├── figures
│   └── reports
└── README.md
```

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/neuromorphic-classification.git
   cd neuromorphic-classification
   ```
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Data preparation

1. Download the NMNIST or DVS Gesture dataset.
2. Update paths in `data/datasets.py` to point to your data directory.
3. The code converts events into 5D voxel tensors of shape `[batch, time, channels, height, width]`.

## Training

### Spiking RNN ROI predictor + SNN

```bash
python scripts/train_snn.py \
  --epochs 5 \
  --batch-size 32 \
  --lr 1e-3 \
  --num-workers 2
```

This script displays batch‑level progress using `tqdm` and saves the best model under `results/`.

### CNN baseline

```bash
python scripts/train_cnn.py \
  --epochs 5 \
  --batch-size 32 \
  --lr 1e-3 \
  --num-workers 2
```

## Evaluation

Run the evaluation script to generate confusion matrices and classification reports:

```bash
python scripts/evaluate.py \
  --model-path results/snn_best.pth \
  --dataset test
```

The script prints a classification report to the console and saves a confusion matrix plot in `results/figures`.

## Results

Training the SNN improved accuracy from around 17% to nearly 68% over five epochs. The confusion matrix highlighted bias towards certain classes, prompting the use of class weights.

## Future work

* Explore alternative ROI predictor architectures
* Add event‑data augmentation strategies
* Investigate online learning and adaptive threshold techniques

## License

This project is licensed under MIT.
