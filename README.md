# Road Detection from Aerial Images

Binary semantic segmentation project (U-Net / ResNet34) to detect roads in aerial images.
This repository is a clean conversion of a Colab notebook to runnable Python scripts.

## Quickstart

1. Create virtualenv and install:
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Download dataset (see scripts/download_data.sh) and place under `data/mass_roads/tiff/`:
```
data/mass_roads/tiff/
  ├── train/
  ├── train_labels/
  ├── val/
  ├── val_labels/
  ├── test/
  └── test_labels/
```

3. Train:
```bash
python -m src.train --data-dir data/mass_roads/tiff --epochs 20 --batch-size 4
```

4. Evaluate:
```bash
python -m src.evaluate --data-dir data/mass_roads/tiff --weights models/checkpoint_epoch20.pth
```

## Notes
- Default image size 256, batch size 4. Adjust with CLI flags.
- Check `models/` for saved checkpoints.
