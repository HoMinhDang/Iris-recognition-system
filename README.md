# Pattern Recognition - Iris Project

This project focuses on iris pattern recognition using the MMU iris dataset. It provides tools for preprocessing, feature extraction, matching, and visualizing results.

---

## Dataset Setup

Download the MMU iris dataset from the following link:

   [Download MMU Dataset](https://drive.google.com/drive/folders/1BUehEn-zLe20MwbbdQsiPfi1sb4TeRrS?usp=sharing)

---

## Environment Setup

Make sure you have [conda](https://docs.conda.io/en/latest/) installed.

Create and activate the environment:
```bash
conda env create -f env.yml
conda activate iris-env
```

> If you need to add packages, update `env.yml` and run `conda env update --file env.yml --prune`.

---

## Run the Project

To start the main program:
```bash
python main.py
```

To run the visualizer (for inspecting iris data and matching results):
```bash
python visualizer.py
```

---

## Enroll & Verify

To enable the verification feature, make sure to run the enrollment script **before** verification:
```bash
python enroll_verify.py
```

---

## Citation

Dataset used in this project:

> **MMU Iris Image Database**  
> Faculty of Engineering, Multimedia University, Malaysia.

---