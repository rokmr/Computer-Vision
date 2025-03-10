import argparse
from src.train import train
from src.inference import VideoPredictor

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description='Cricket Shot Classification')
    
    # Add arguments
    parser.add_argument(
        'mode',
        type=str,
        choices=['train', 'inference'],
        help='Mode to run the model in (train or inference)'
    )
    
    # Training arguments
    parser.add_argument(
        '--model_ckpt',
        type=str,
        default="MCG-NJU/videomae-base",
        help='Model checkpoint to start from'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-5,
        help='Learning rate'
    )
    
    # Inference arguments
    parser.add_argument(
        '--video_path',
        type=str,
        help='Path to video file for inference'
    )
    parser.add_argument(
        '--save_gif',
        action='store_true',
        help='Save processed frames as GIF'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to trained model for inference'
    )

    args = parser.parse_args()

    if args.mode == 'train':
        # Run training
        print("Starting training...")
        train_results = train(
            model_ckpt=args.model_ckpt,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        print("Training completed!")
        print("Results:", train_results)
        
    else:  # inference mode
        if not args.video_path:
            raise ValueError("Video path must be provided for inference mode")
        if not args.model_path:
            raise ValueError("Model path must be provided for inference mode")
            
        print(f"Running inference on video: {args.video_path}")
        
        # Initialize predictor
        predictor = VideoPredictor(args.model_path)
        
        # Run prediction
        results = predictor.predict(
            video_path=args.video_path,
            save_gif=args.save_gif
        )
        
        # Print results
        print("\nPrediction Results:")
        print(f"Predicted Class: {results['predicted_class']}")
        print(f"Confidence: {results['confidence']:.2%}")
        if results['gif_path']:
            print(f"GIF saved to: {results['gif_path']}")

if __name__ == "__main__":
    main()
