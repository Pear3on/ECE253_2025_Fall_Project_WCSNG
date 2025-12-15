## Dataset

The dataset used in this project is provided in the `dataset` directory.

```bash
cd dataset
```

The first 170 images are used as the test set, while the remaining images are used as the validation set.

Dataset Link: \href{https://drive.google.com/file/d/1pOdMrvXrhLBH4s2rnYgf5hxx1BbhwHWH/view?usp=drive_link}{Google Drive}

## Image Processing Pipeline

All image enhancement and motion blur removal steps are implemented under the `Image_process` directory.

```bash
cd Image_process
```
Please follow the execution steps described in `pipeline.ipynb` to run the full preprocessing pipeline.

## Evaluation

Quantitative evaluation of image quality metrics is conducted under the `Image_process` directory.

```bash
cd Image_process
```

Follow the instructions in `eval.ipynb` to reproduce the evaluation results.


## VLM Object Detection:

### First Step:

Download model weights and put it under the `GroundingDINO/weights` directory.

```bash
cd GroundingDINO/weights
```

Link is here: \href{https://drive.google.com/file/d/1Wq0T5Rt0_ARCXOQLJ6oVvO1oSf_u8Kzy/view?usp=drive_link}{Google Drive}

### Second Step:

Vision–language model based object detection experiments are implemented using Grounding DINO.

```bash
cd GroundingDINO
```

Follow the steps in `test.ipynb` to run VLM-based object detection and obtain detection outputs.


