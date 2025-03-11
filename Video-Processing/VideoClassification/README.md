# Video Classification Project

This project implements video classification using two different approaches:
1. Using LSTM networks on top of pre-trained image embeddings (CLIP/SigLIP)
2. Fine-tuning VideoMAE for end-to-end video classification


## Features

- Video dataset processing and frame extraction
- Frame embedding generation using CLIP or SigLIP models
- LSTM-based classification using pre-trained embeddings
- VideoMAE fine-tuning for direct video classification
- Comprehensive logging system
- Support for model pushing to HuggingFace Hub


## Installation

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate cricshot
```
NOTE: Make sure to huggingface token in the `.env` file

## Project Structure

```
VideoClassification/
├── src/
│   ├── data.py           # Dataset processing utilities
│   ├── embed_lstm.py     # LSTM-based classification
│   ├── finetune_videomae.py  # VideoMAE fine-tuning
│   └── logger.py         # Logging configuration
├── data/                 # Dataset storage
│   └── frames/          # Extracted video frames
├── logs/                # Log files
├── environment.yml      # Conda environment specification
└── README.md
```

## Dataset 
For this project athe dataset is created and published at [rokmr/cricketshot-predictor](https://huggingface.co/datasets/rokmr/cricket-shot) on huggingface 🤗.

## Usage

### 1. Dataset Processing

First, process the video dataset to extract frames:

```bash
python src/data.py
```

This will:
- Download the dataset from HuggingFace Hub
- Extract video frames
- Create necessary directory structure
- Generate label mappings

### 2. LSTM-based Classification
**CLIP**
```python
python src/embed_lstm.py --model_id "openai/clip-vit-base-patch32" \
                        --embed_size 512  \
                        --save_model_path "./clip-cricket-classifier.pt" \
                        --create_embeddings
```

**SigLIP**

```python
python src/embed_lstm.py --model_id "google/siglip-base-patch16-224" \
                        --embed_size 768  \
                        --save_model_path "./siglip-cricket-classifier.pt" \
                        --create_embeddings 
```

This will:
- Extract visual features using the CLIP/SigLIP model
- Train an LSTM network using the extracted features
- Save the model checkpoint with highest validation accuracy
- Evaluate model performance on the test dataset and generate metrics

### 3. VideoMAE-finetuning based Classification

Train the VideoMAE model for video classification:

```bash
python src/finetune_videomae.py --model_ckpt "MCG-NJU/videomae-base" \
                                --output_dir "cricketshot-predictor" \
                                --num_epochs 10 \
                                --batch_size 2 \
                                --learning_rate 5e-5
```

Additional options:
- `--push_to_hub`: Push the trained model to HuggingFace Hub
- `--test_only`: Evaluate a trained model without training

Example for testing only:
```bash
python src/finetune_videomae.py --model_ckpt "rokmr/cricketshot-predictor" \
                               --test_only
```
Replace `rokmr` with your huggingface `user_id`
This will:
- Fine-tune the VideoMAE model on the cricket shots dataset
- Save model checkpoints after each epoch
- Evaluate model performance on validation set during training
- Generate final metrics on the test dataset
- Optionally push the model to HuggingFace Hub

## Results

| Method | Test Accuracy |
|--------|---------------|
| CLIP + LSTM | 53% |
| SigLIP + LSTM | 50% |
| VideoMAE | 66.05% |

## Acknowledgments

- VideoMAE model from MCG-NJU
- CLIP model from OpenAI
- SigLIP model from Google