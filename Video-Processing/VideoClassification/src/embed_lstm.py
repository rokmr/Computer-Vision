import os
import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from PIL import Image
import dotenv
from transformers import AutoModel, AutoProcessor

from logger import logger
from data import VideoDatasetProcessor

dotenv.load_dotenv()
torch.manual_seed(42)


class Embedder:       
    def __init__(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = args.model_id
        try:
            logger.info(f"Loading model: {self.model_id}")
            self.model = AutoModel.from_pretrained(self.model_id)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        self.embed_size = args.embed_size
        self.dataset_root_path = args.dataset_root_path
        self.frame_dir = args.frame_dir
        self.base_path = args.base_path
        self._make_output_dirs()

    def _make_output_dirs(self):
        self.embeddings_folder = f"{self.base_path}/{self.model_id}/frames_embeddings"
        os.makedirs(self.embeddings_folder, exist_ok=True)

    def _create_embeddings(self):
        try:
            logger.info(f"Creating embeddings for {self.model_id}")
            splits = ['train', 'val', 'test']
            for split in splits:
                split_folder = os.path.join(self.frame_dir, split)
                output_split_folder = os.path.join(self.embeddings_folder, split)
                os.makedirs(output_split_folder, exist_ok=True)

                class_folders = [f.path for f in os.scandir(split_folder) if f.is_dir()]
                processed_classes = 0

                for class_folder in tqdm(class_folders, desc=f"Processing {split} split"):
                    class_name = os.path.basename(class_folder)
                    output_class_folder = os.path.join(output_split_folder, class_name)
                    os.makedirs(output_class_folder, exist_ok=True)

                    image_files = [f for f in os.listdir(class_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]    

                    if image_files:
                        image_paths = [os.path.join(class_folder, f) for f in image_files]
                        embeddings = self._batch_process_images(image_paths, batch_size=100)

                        for i, emb in enumerate(embeddings):
                            original_file_name = image_files[i].rsplit('.', 1)[0]
                            output_path = os.path.join(output_class_folder, f'{original_file_name}_embedding.npy')
                            np.save(output_path, emb)

                        processed_classes += 1
        except Exception as e:
            logger.error(f"Error while creating embeddings: {e}")
            raise

    def _batch_process_images(self, image_paths, batch_size):
        embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = [Image.open(path).convert("RGB") for path in batch_paths]
            tokens = self.processor(
                text=None,
                images=batch_images,
                return_tensors="pt"
            ).to(self.device)
            batch_embeddings = self.model.get_image_features(**tokens)
            batch_embeddings = batch_embeddings.detach().cpu().numpy()
            embeddings.append(batch_embeddings)
        return np.concatenate(embeddings, axis=0)
    
    def _load_embeddings_and_labels(self, embeddings_folder):
        try:
            logger.info(f"Loading embeddings and labels for {embeddings_folder}")
            embeddings = []
            labels = []
            label_mapping = {}  # To convert class names to numerical labels
            current_label = 0
            class_folders = [f.path for f in os.scandir(embeddings_folder) if f.is_dir()]
            logger.info(f"Found {len(class_folders)} classes")
            for class_folder in class_folders:
                if os.path.isdir(class_folder):
                    if class_folder not in label_mapping:
                        class_name = os.path.basename(class_folder)
                        label_mapping[class_name] = current_label
                        current_label += 1
                    for emb_file in sorted(os.listdir(class_folder)):
                        if emb_file.endswith('_embedding.npy'):
                            emb_path = os.path.join(class_folder, emb_file)
                            embeddings.append(np.load(emb_path))
                            labels.append(label_mapping[class_name])
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            return embeddings, labels, label_mapping
        except Exception as e:
            logger.error(f"Error while loading embeddings and labels: {e}")
            raise
    
    def _create_datasets(self):
        train_embeddings, train_labels, class_label_mapping = self._load_embeddings_and_labels(os.path.join(self.embeddings_folder, 'train'))
        val_embeddings, val_labels, _ = self._load_embeddings_and_labels(os.path.join(self.embeddings_folder, 'val')) 
        test_embeddings, test_labels, _ = self._load_embeddings_and_labels(os.path.join(self.embeddings_folder, 'test'))
        logger.info(f"Train embeddings shape: {train_embeddings.shape} || Val embeddings shape: {val_embeddings.shape} || Test embeddings shape: {test_embeddings.shape}")

        try:
            train_dataset = TensorDataset(train_embeddings.unsqueeze(1), train_labels)
            val_dataset = TensorDataset(val_embeddings.unsqueeze(1), val_labels)
            test_dataset = TensorDataset(test_embeddings.unsqueeze(1), test_labels)
            return train_dataset, val_dataset, test_dataset, class_label_mapping
        except Exception as e:
            logger.error(f"Error while creating datasets: {e}")
            raise

class LSTMNetwork(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_classes=4):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  
        return x

def train_model(model, train_loader, val_loader, optimizer, criterion, device, args):
    logger.info(f"Training model for {args.num_epochs} epochs")
    num_epochs = args.num_epochs
    max_val_acc = 0
    max_val_acc_epoch = 0
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        total_correct = total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                total_correct += (predicted == target).sum().item()
                total += target.size(0)

        val_accuracy = total_correct / total
        logger.info(f"Epoch {epoch + 1} || Training Loss: {total_loss / len(train_loader):.6f} || Val Acc.: {val_accuracy:.2f}")

        if val_accuracy > max_val_acc:
            logger.info(f"Saving model at epoch {epoch + 1} with val accuracy: {val_accuracy:.2f}")
            torch.save(model.state_dict(), args.save_model_path)
            max_val_acc = max(max_val_acc, val_accuracy)
            max_val_acc_epoch = epoch + 1

    logger.info(f"Best Model Saved {args.save_model_path}: Max Val Acc. {max_val_acc:.2f} at epoch {max_val_acc_epoch}")


def test_model(model, test_loader, class_label_mapping, device):
    logger.info(f"Testing model")
    model.eval()
    total_correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == target).sum().item()
            total += target.size(0)
            all_labels.extend(target.tolist())
            all_preds.extend(predicted.tolist())

    test_accuracy = total_correct / total
    logger.info(f"Test Acc.: {test_accuracy:.2f}")
    conf_mat = confusion_matrix(all_labels, all_preds)
    logger.info("Confusion Matrix:")
    logger.info(conf_mat)

    report = classification_report(all_labels, all_preds, target_names=list(class_label_mapping.keys()))
    logger.info("Classification Report:")
    logger.info(report)


def main(args):
    embedder = Embedder(args)
    if args.create_embeddings:
        embedder._create_embeddings()
    train_dataset, val_dataset, test_dataset, class_label_mapping = embedder._create_datasets()
    device = embedder.device
    logger.info(f"Loading data: training samples: {len(train_dataset)} || validation samples: {len(val_dataset)} || test samples: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    logger.info(f"Initializing LSTM Network: input size: {args.embed_size} || hidden size: 256 || number of classes: {len(class_label_mapping)}")
    model = LSTMNetwork(input_size=args.embed_size, hidden_size=256, num_classes=len(class_label_mapping)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_model(model, train_loader, val_loader, optimizer, criterion, device, args)
    test_model(model, test_loader, class_label_mapping, device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/siglip-base-patch16-224") #google/siglip-base-patch16-224, openai/clip-vit-base-patch32
    parser.add_argument("--embed_size", type=int, default=768) #768, 512
    parser.add_argument("--create_embeddings", action="store_true", help="Create new embeddings")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_model_path", type=str, default="./siglip-cricket-classifier.pt") 
    parser.add_argument("--num_epochs", type=int, default=50)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    video_data = VideoDatasetProcessor()

    logger.info(f"Args: {args}")
    args.dataset_root_path = video_data.dataset_root_path
    args.frame_dir = video_data.frame_dir
    args.base_path = video_data.base_path
    main(args)