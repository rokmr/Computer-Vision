import os
import numpy as np
import torch
import subprocess
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import shutil
import requests
from tqdm import tqdm
import zipfile
import pathlib
import tarfile
import dotenv
from huggingface_hub import hf_hub_download
from transformers import CLIPProcessor, CLIPModel
dotenv.load_dotenv()


hf_dataset_identifier = "rokmr/cricket-shot"
filename = "cricketshot.tar.gz"
file_path = hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset")

with tarfile.open(file_path) as t:
    # Extract all files
    t.extractall(".")
    # Get the extraction directory
    extract_dir = "."
    # Remove any files starting with "._"
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.startswith("._"):
                os.remove(os.path.join(root, file))
    

dataset_root_path = pathlib.Path('cricketshot')
video_count_train = len(list(dataset_root_path.glob("train/*/*.avi")))
video_count_val = len(list(dataset_root_path.glob("val/*/*.avi")))
video_count_test = len(list(dataset_root_path.glob("test/*/*.avi")))
video_total = video_count_train + video_count_val + video_count_test
print(f"Total videos: {video_total}")
print(f"Train videos: {video_count_train}")
print(f"Val videos: {video_count_val}")
print(f"Test videos: {video_count_test}")

all_video_file_paths = (
    list(dataset_root_path.glob("train/*/*.avi"))
    + list(dataset_root_path.glob("val/*/*.avi"))
    + list(dataset_root_path.glob("test/*/*.avi"))
)
print(f"Total videos: {len(all_video_file_paths)}")
print(f"First 5 videos: {all_video_file_paths[:5]}")


base_path = './data'
os.makedirs(base_path, exist_ok=True)

frame_dir = f"{base_path}/frames"
os.makedirs(frame_dir, exist_ok=True)


#Setting up FFMPEG PATH
FFMPEG_PATH = shutil.which("ffmpeg")

if FFMPEG_PATH is None:
    print("Error: ffmpeg not found")
else:
    print(f"ffmpeg located at: {FFMPEG_PATH}")




# Iterate over train, val, test folders
for split in ['train', 'val', 'test']:
    split_path = dataset_root_path / split
    
    # Create output directory for this split
    split_output_dir = os.path.join(frame_dir, split)
    os.makedirs(split_output_dir, exist_ok=True)

    # Iterate over each class folder in this split
    for class_folder in os.listdir(split_path):
        class_folder_path = split_path / class_folder
        
        # Ensure it's a directory
        if os.path.isdir(class_folder_path):
            # Create an output folder for each class under the split
            class_output_dir = os.path.join(split_output_dir, class_folder)
            os.makedirs(class_output_dir, exist_ok=True)

            # Iterate over each video file in the class folder
            for video_file in os.listdir(class_folder_path):
                if video_file.endswith('.avi'):
                    video_path = class_folder_path / video_file
                    video_name = os.path.splitext(video_file)[0]
                    output_pattern = os.path.join(class_output_dir, f"{video_name}_%04d.jpg")
                    
                    # Command to extract frames at 1 fps
                    ffmpeg_command = f"{FFMPEG_PATH} -i {video_path} -vf fps=1 {output_pattern} -loglevel quiet"
                    
                    # Execute the command
                    try:
                        subprocess.run(ffmpeg_command, shell=True, check=True)
                        print(f"Processed {video_file} in {split}/{class_folder}")
                    except subprocess.CalledProcessError:
                        print(f"Failed to process video: {video_file} in {split}/{class_folder}")

# Load the CLIP model and processor
model_id = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

def batch_process_images(image_paths, batch_size, processor, model, device):
    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = [Image.open(path).convert("RGB") for path in batch_paths]
        tokens = processor(
            text=None,
            images=batch_images,
            return_tensors="pt"
        ).to(device)
        batch_embeddings = model.get_image_features(**tokens)
        batch_embeddings = batch_embeddings.detach().cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.concatenate(embeddings, axis=0)

# Path where the extracted frames are stored
main_folder = f"{base_path}/frames"
output_folder = f"{base_path}/frames_embeddings"
os.makedirs(output_folder, exist_ok=True)

# Process train, val and test splits
for split in ['train', 'val', 'test']:
    split_folder = os.path.join(main_folder, split)
    output_split_folder = os.path.join(output_folder, split)
    os.makedirs(output_split_folder, exist_ok=True)
    
    # Get all class folders in this split
    class_folders = [f.path for f in os.scandir(split_folder) if f.is_dir()]
    total_classes = len(class_folders)
    processed_classes = 0

    for class_folder in class_folders:
        class_name = os.path.basename(class_folder)
        output_class_folder = os.path.join(output_split_folder, class_name)
        os.makedirs(output_class_folder, exist_ok=True)
        
        image_files = [f for f in os.listdir(class_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            image_paths = [os.path.join(class_folder, f) for f in image_files]
            embeddings = batch_process_images(image_paths, batch_size=100, processor=processor, model=clip_model, device=device)

            # Save each embedding with a filename that reflects its original image
            for i, emb in enumerate(embeddings):
                original_file_name = image_files[i].rsplit('.', 1)[0]  # Remove extension
                output_path = os.path.join(output_class_folder, f'{original_file_name}_embedding.npy')
                np.save(output_path, emb)

        processed_classes += 1
        print(f"Processed {processed_classes}/{total_classes} classes in {split} split.")

    print(f"Completed processing {split} split.")

print("All splits processed.")


# Load embeddings and their labels
def load_embeddings_and_labels(embeddings_folder):
    embeddings = []
    labels = []
    label_mapping = {}  # To convert class names to numerical labels
    current_label = 0

    for class_folder in sorted(os.listdir(embeddings_folder)):
        class_path = os.path.join(embeddings_folder, class_folder)
        if os.path.isdir(class_path):
            if class_folder not in label_mapping:
                label_mapping[class_folder] = current_label
                current_label += 1
            for emb_file in sorted(os.listdir(class_path)):
                if emb_file.endswith('_embedding.npy'):
                    emb_path = os.path.join(class_path, emb_file)
                    embeddings.append(np.load(emb_path))
                    labels.append(label_mapping[class_folder])

    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    return embeddings, labels, label_mapping

# Define the LSTM neural network
class LSTMNetwork(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_classes=4):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Use the output of the last time step
        return x
    

# Load data from train/val/test splits
embeddings_folder = f"{base_path}/frames_embeddings"

# Load embeddings and labels for each split
train_embeddings, train_labels, class_label_mapping = load_embeddings_and_labels(os.path.join(embeddings_folder, 'train'))
val_embeddings, val_labels, _ = load_embeddings_and_labels(os.path.join(embeddings_folder, 'val')) 
test_embeddings, test_labels, _ = load_embeddings_and_labels(os.path.join(embeddings_folder, 'test'))

# Create datasets
train_dataset = TensorDataset(train_embeddings.unsqueeze(1), train_labels)
val_dataset = TensorDataset(val_embeddings.unsqueeze(1), val_labels)
test_dataset = TensorDataset(test_embeddings.unsqueeze(1), test_labels)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model
model = LSTMNetwork(input_size=512, hidden_size=256, num_classes=len(class_label_mapping)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for data, target in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader)}")

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
    print(f"Validation Accuracy: {val_accuracy:.2f}")

# Test
model.eval()
total_correct = total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output, 1)
        total_correct += (predicted == target).sum().item()
        total += target.size(0)

test_accuracy = total_correct / total
print(f"Test Accuracy: {test_accuracy:.2f}")

#confusion matrix
all_labels = []
all_preds = []
for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    output = model(data)
    _, predicted = torch.max(output, 1)
    all_labels.extend(target.tolist())
    all_preds.extend(predicted.tolist())

conf_mat = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(conf_mat)

from sklearn.metrics import classification_report
report = classification_report(all_labels, all_preds, target_names=list(class_label_mapping.keys()))
print("Classification Report:")
print(report)


# Save model
filepath = "./cricket.pt"
torch.save(model.state_dict(), filepath)