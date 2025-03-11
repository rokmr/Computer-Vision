import tarfile
import pathlib
import os
import shutil
import subprocess
from huggingface_hub import hf_hub_download
import dotenv
from logger import logger

dotenv.load_dotenv()

class VideoDatasetProcessor:
    def __init__(self, dataset_id="rokmr/cricket-shot", filename="cricketshot.tar.gz"):
        logger.info(f"Initializing VideoDatasetProcessor with dataset_id={dataset_id}, filename={filename}")
        self.dataset_id = dataset_id
        self.filename = filename
        self.dataset_root_path = pathlib.Path('cricketshot')
        self.base_path = './data'
        self.frame_dir = f"{self.base_path}/frames"
        self.ffmpeg_path = shutil.which("ffmpeg")
        if self.ffmpeg_path is None:
            logger.error("ffmpeg not found in system PATH")
        
        self._setup_directories()
        self._initialize_label_mappings()
        
    def _setup_directories(self):
        """Create necessary directories for data processing"""
        logger.info("Creating directories for data processing")
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.frame_dir, exist_ok=True)
        logger.debug(f"Created directories: {self.base_path}, {self.frame_dir}")
        
    def _initialize_label_mappings(self):
        """Initialize label to index and index to label mappings"""
        train_path = self.dataset_root_path / "train"
        if train_path.exists():
            self.classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
            self.label_to_idx = {label: idx for idx, label in enumerate(self.classes)}
            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
            logger.info(f"Found {len(self.classes)} classes: {', '.join(self.classes)}")
        else:
            logger.warning(f"Train path {train_path} does not exist. Label mappings will be empty.")
            self.classes = []
            self.label_to_idx = {}
            self.idx_to_label = {}
    
    def get_label_mapping(self):
        """Return the label to index mapping dictionary"""
        return self.label_to_idx
    
    def get_index_mapping(self):
        """Return the index to label mapping dictionary"""
        return self.idx_to_label
    
    def download_and_extract(self):
        """Download and extract the dataset"""
        logger.info(f"Downloading dataset from {self.dataset_id}")
        try:
            file_path = hf_hub_download(
                repo_id=self.dataset_id, 
                filename=self.filename, 
                repo_type="dataset"
            )
            
            with tarfile.open(file_path) as t:
                logger.info(f"Extracting {file_path} to {os.getcwd()}")
                t.extractall(".")
                
                # Remove hidden files
                for root, _, files in os.walk("."):
                    for file in files:
                        if file.startswith("._"):
                            file_path = os.path.join(root, file)
                            logger.debug(f"Removing hidden file: {file_path}")
                            os.remove(file_path)
            logger.info("Dataset download and extraction completed successfully")
        except Exception as e:
            logger.error(f"Failed to download or extract dataset: {str(e)}")
            raise
    
    def get_video_statistics(self):
        """Get count of videos in each split"""
        logger.info("Calculating video statistics")
        counts = {
            'train': len(list(self.dataset_root_path.glob("train/*/*.avi"))),
            'val': len(list(self.dataset_root_path.glob("val/*/*.avi"))),
            'test': len(list(self.dataset_root_path.glob("test/*/*.avi")))
        }
        counts['total'] = sum(counts.values())
        logger.info(f"Found {counts['total']} total videos: {counts['train']} train, {counts['val']} val, {counts['test']} test")
        return counts
    
    def get_all_video_paths(self):
        """Get paths of all videos in the dataset"""
        return (
            list(self.dataset_root_path.glob("train/*/*.avi")) +
            list(self.dataset_root_path.glob("val/*/*.avi")) +
            list(self.dataset_root_path.glob("test/*/*.avi"))
        )
    
    def extract_frames(self):
        """Extract frames from all videos using ffmpeg"""
        logger.info(f"Extracting frames from all videos using ffmpeg")
        if self.ffmpeg_path is None:
            raise RuntimeError("Error: ffmpeg not found")
            
        for split in ['train', 'val', 'test']:
            split_path = self.dataset_root_path / split
            split_output_dir = os.path.join(self.frame_dir, split)
            os.makedirs(split_output_dir, exist_ok=True)
            
            for class_folder in os.listdir(split_path):
                class_folder_path = split_path / class_folder
                
                if os.path.isdir(class_folder_path):
                    class_output_dir = os.path.join(split_output_dir, class_folder)
                    os.makedirs(class_output_dir, exist_ok=True)
                    
                    for video_file in os.listdir(class_folder_path):
                        if video_file.endswith('.avi'):
                            self._process_video(
                                video_file, 
                                class_folder_path, 
                                class_output_dir, 
                                split, 
                                class_folder
                            )
    
    def _process_video(self, video_file, class_folder_path, class_output_dir, split, class_folder):
        """Process individual video file to extract frames"""
        video_path = class_folder_path / video_file
        video_name = os.path.splitext(video_file)[0]
        output_pattern = os.path.join(class_output_dir, f"{video_name}_%04d.jpg")
        
        ffmpeg_command = f"{self.ffmpeg_path} -i {video_path} -vf fps=5 {output_pattern} -loglevel quiet"
        
        try:
            subprocess.run(ffmpeg_command, shell=True, check=True)
            logger.info(f"Successfully processed {video_file} in {split}/{class_folder}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to process video {video_file} in {split}/{class_folder}: {str(e)}")
            raise

def main():
    try:
        processor = VideoDatasetProcessor()
        processor.download_and_extract()
        stats = processor.get_video_statistics()
        processor.extract_frames()
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()