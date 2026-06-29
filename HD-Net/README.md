# HD-Net

PyTorch code for cuffless blood pressure estimation from ECG and PPG signals.

This GitHub-ready version keeps the original project code files and does not add newly split HD-Net code files. Plotting scripts, generated figures, data files, HD-Net weights, caches, and experiment outputs are excluded.

## Files

```text
HD-Net/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── Evaluate_TTA.py
├── Model_Training/
│   ├── Model_Training.py
│   └── Model_Def/
│       ├── ResNet.py
│       └── Trainer.py
├── tools/
│   ├── convert_mat_to_npz.py
│   ├── convert_mat_to_npz_deep.py
│   └── split_train_val_npz.py
└── data/
    └── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA environment if needed.

## Dataset

Download the source dataset and prepare NPZ files under `data/hdnet/`. See `data/README.md` for expected filenames and keys.

## Training

```bash
python Model_Training/Model_Training.py --data-folder data/hdnet --targets SBP DBP
```

Train one target:

```bash
python Model_Training/Model_Training.py --data-folder data/hdnet --targets SBP
python Model_Training/Model_Training.py --data-folder data/hdnet --targets DBP
```

## Evaluation

Place trained HD-Net weights at:

```text
checkpoints/SBP/trained_model.pth
checkpoints/DBP/trained_model.pth
```

Then run:

```bash
python Evaluate_TTA.py
```

The included training/evaluation code writes numeric logs only. Plot-generation scripts and figure outputs are not included.




