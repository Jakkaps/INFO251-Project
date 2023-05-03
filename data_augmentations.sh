rm data_augmentations.py
jupyter nbconvert data_augmentations.ipynb --log-level=WARN --to script && python data_augmentations.py
