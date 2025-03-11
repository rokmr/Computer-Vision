"""
VideoMAE fine-tuning for video classification.

This module implements fine-tuning of VideoMAE models for video classification tasks.
"""

import os
import pathlib
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import imageio
import numpy as np
import pytorchvideo.data
import torch
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

from transformers import TrainingArguments, Trainer
import evaluate
import dotenv
from huggingface_hub import hf_hub_download
import tarfile
import argparse
from data import VideoDatasetProcessor
from logger import setup_logger, get_log_file

# Initialize logger
logger = setup_logger(__name__, get_log_file('videomae'))
dotenv.load_dotenv()

class VideoMAETrainer:
    """Handles the training of VideoMAE models for video classification tasks."""

    def __init__(self, args):
        """Initialize the trainer with command line arguments."""
        self.args = args
        logger.info("Initializing VideoMAETrainer with args: %s", vars(args))
        
        self.dataset_processor = VideoDatasetProcessor()
        self.setup_model()
        self.setup_transforms()
        self.setup_datasets()
        self.metric = evaluate.load("accuracy")
        logger.info("VideoMAETrainer initialization completed")

    def setup_model(self):
        """Initialize the model and image processor."""
        logger.info("Setting up model with checkpoint: %s", self.args.model_ckpt)
        try:
            self.label2id = self.dataset_processor.get_label_mapping()
            self.id2label = self.dataset_processor.get_index_mapping()
            logger.debug("Label mappings loaded: %d classes", len(self.label2id))
            
            self.image_processor = VideoMAEImageProcessor.from_pretrained(self.args.model_ckpt)
            self.model = VideoMAEForVideoClassification.from_pretrained(
                self.args.model_ckpt,
                label2id=self.label2id,
                id2label=self.id2label,
                ignore_mismatched_sizes=True,
            )
            logger.info("Model and image processor loaded successfully")
        except Exception as e:
            logger.error("Error setting up model: %s", str(e))
            raise

    def setup_transforms(self):
        """Set up video transformation pipelines for training and validation."""
        logger.info("Setting up video transforms")
        try:
            # Get image processing parameters
            mean = self.image_processor.image_mean
            std = self.image_processor.image_std
            
            # Determine resize dimensions
            if "shortest_edge" in self.image_processor.size:
                height = width = self.image_processor.size["shortest_edge"]
            else:
                height = self.image_processor.size["height"]
                width = self.image_processor.size["width"]
            resize_to = (height, width)
            logger.debug("Resize dimensions: %s", resize_to)

            # Calculate clip duration
            num_frames_to_sample = self.model.config.num_frames
            sample_rate = 4
            fps = 30
            self.clip_duration = num_frames_to_sample * sample_rate / fps
            logger.debug("Clip duration: %.2f seconds", self.clip_duration)

            # Create transforms
            self.train_transform = self._create_train_transform(num_frames_to_sample, mean, std, resize_to)
            self.val_transform = self._create_val_transform(num_frames_to_sample, mean, std, resize_to)
            logger.info("Video transforms created successfully")
        except Exception as e:
            logger.error("Error setting up transforms: %s", str(e))
            raise

    def _create_train_transform(self, num_frames, mean, std, resize_to):
        """Create transformation pipeline for training data."""
        logger.debug("Creating training transforms with num_frames=%d", num_frames)
        return Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(resize_to),
                    RandomHorizontalFlip(p=0.5),
                ]),
            ),
        ])

    def _create_val_transform(self, num_frames, mean, std, resize_to):
        """Create transformation pipeline for validation data."""
        logger.debug("Creating validation transforms with num_frames=%d", num_frames)
        return Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to),
                ]),
            ),
        ])

    def setup_datasets(self):
        """Initialize training and validation datasets."""
        logger.info("Setting up datasets")
        try:
            dataset_root_path = self.dataset_processor.dataset_root_path
            
            logger.debug("Creating training dataset")
            self.train_dataset = pytorchvideo.data.Ucf101(
                data_path=os.path.join(dataset_root_path, "train"),
                clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.clip_duration),
                decode_audio=False,
                transform=self.train_transform,
            )

            logger.debug("Creating validation dataset")
            self.val_dataset = pytorchvideo.data.Ucf101(
                data_path=os.path.join(dataset_root_path, "val"),
                clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
                decode_audio=False,
                transform=self.val_transform,
            )

            logger.debug("Creating test dataset")
            self.test_dataset = pytorchvideo.data.Ucf101(
                data_path=os.path.join(dataset_root_path, "test"),
                clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
                decode_audio=False,
                transform=self.val_transform,
            )
            
            logger.info("Datasets created - Train: %d videos, Val: %d videos, Test: %d videos",
                       self.train_dataset.num_videos,
                       self.val_dataset.num_videos,
                       self.test_dataset.num_videos)
        except Exception as e:
            logger.error("Error setting up datasets: %s", str(e))
            raise

    def compute_metrics(self, eval_pred):
        """Compute accuracy metrics for evaluation."""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        metrics = self.metric.compute(predictions=predictions, references=eval_pred.label_ids)
        logger.debug("Computed metrics: %s", metrics)
        return metrics

    @staticmethod
    def collate_fn(examples):
        """Collate function for batching data."""
        pixel_values = torch.stack([example["video"].permute(1, 0, 2, 3) for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    def train(self):
        """Execute the training process."""
        logger.info("Starting training process")
        try:
            training_args = TrainingArguments(
                self.args.output_dir,
                remove_unused_columns=False,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=self.args.learning_rate,
                per_device_train_batch_size=self.args.batch_size,
                per_device_eval_batch_size=self.args.batch_size,
                warmup_ratio=0.1,
                logging_steps=10,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                push_to_hub=self.args.push_to_hub,
                max_steps=(self.train_dataset.num_videos // self.args.batch_size) * self.args.num_epochs,
            )

            logger.info("Training configuration: %s", {
                'num_epochs': self.args.num_epochs,
                'batch_size': self.args.batch_size,
                'learning_rate': self.args.learning_rate,
                'max_steps': training_args.max_steps,
            })

            trainer = Trainer(
                self.model,
                training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                tokenizer=self.image_processor,
                compute_metrics=self.compute_metrics,
                data_collator=self.collate_fn,
            )

            train_results = trainer.train()
            logger.info("Training completed. Results: %s", train_results)
            
            # Evaluate on test set
            test_results = trainer.evaluate(self.test_dataset)
            logger.info("Test Set Results: %s", test_results)
            
            if self.args.push_to_hub:
                logger.info("Pushing model to HuggingFace Hub")
                trainer.push_to_hub()
                logger.info("Model successfully pushed to hub")
        except Exception as e:
            logger.error("Error during training: %s", str(e))
            raise

    def test(self):
        """Execute testing on a trained model."""
        logger.info("Starting testing process")
        try:
            training_args = TrainingArguments(
                self.args.output_dir,
                remove_unused_columns=False,
                per_device_eval_batch_size=self.args.batch_size,
            )

            trainer = Trainer(
                self.model,
                training_args,
                tokenizer=self.image_processor,
                compute_metrics=self.compute_metrics,
                data_collator=self.collate_fn,
            )

            # Evaluate on test set
            test_results = trainer.evaluate(self.test_dataset)
            logger.info("Test Set Results: %s", test_results)
        except Exception as e:
            logger.error("Error during testing: %s", str(e))
            raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune VideoMAE model")
    parser.add_argument("--model_ckpt", default="MCG-NJU/videomae-base", help="Model checkpoint to use")
    parser.add_argument("--output_dir", default="cricketshot-predictor", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--push_to_hub", action="store_true", default=False, help="Push model to HuggingFace Hub")
    parser.add_argument("--test_only", action="store_true", help="Only run testing on trained model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("Starting VideoMAE fine-tuning with args: %s", vars(args))
    
    try:
        trainer = VideoMAETrainer(args)
        if args.test_only:
            trainer.test()
        else:
            trainer.train()
        logger.info("VideoMAE fine-tuning completed successfully")
    except Exception as e:
        logger.error("Error in VideoMAE fine-tuning: %s", str(e))
        raise