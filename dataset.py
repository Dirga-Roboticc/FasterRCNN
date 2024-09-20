import torch
import cv2
import numpy as np
import os
import glob as glob
from xml.etree import ElementTree as et
from config import CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE, NUM_CLASSES
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform

class CustomDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        
        # Check if directory exists
        if not os.path.exists(self.dir_path):
            raise ValueError(f"Directory does not exist: {self.dir_path}")
        
        # Get all image paths (JPG and PNG)
        self.image_paths = glob.glob(f"{self.dir_path}/images/*.jpg") + glob.glob(f"{self.dir_path}/images/*.png") + glob.glob(f"{self.dir_path}/images/*.jpeg")
        if not self.image_paths:
            raise ValueError(f"No jpg, jpeg, or png images found in directory: {self.dir_path}/images")
        
        self.all_images = [os.path.basename(image_path) for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)
        
        print(f"Found {len(self.all_images)} images in {self.dir_path}/images")

    def __getitem__(self, idx):
        # Get image name and path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, 'images', image_name)

        # Read image
        image = cv2.imread(image_path)
        # Convert color format from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        
        # Get XML file for annotation
        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.dir_path, 'Annotations', annot_filename)
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # Mendapatkan ukuran asli gambar
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        # Menyesuaikan koordinat bounding box dengan ukuran gambar yang diberikan
        for member in root.findall('object'):
            class_name = member.find('name').text
            if class_name in self.classes:
                labels.append(self.classes.index(class_name))
            else:
                print(f"Warning: Class '{class_name}' not found in CLASSES. Skipping this object.")
                continue
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            
            xmin_final = (xmin / image_width) * self.width
            xmax_final = (xmax / image_width) * self.width
            ymin_final = (ymin / image_height) * self.height
            ymax_final = (ymax / image_height) * self.height
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        # Mengubah bounding box menjadi tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Membuat dictionary target untuk anotasi
        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([idx])
        }

        # Menerapkan transformasi jika ada
        if self.transforms:
            sample = self.transforms(image=image_resized, 
                                     bboxes=target['boxes'].tolist(), 
                                     labels=labels.tolist())
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            labels = torch.tensor(sample['labels'])
        
        target['labels'] = labels
        
        return image_resized, target

    def __len__(self):
        return len(self.all_images)

# Debugging information
print(f"TRAIN_DIR: {TRAIN_DIR}")
print(f"VALID_DIR: {VALID_DIR}")
print(f"TRAIN_DIR exists: {os.path.exists(TRAIN_DIR)}")
print(f"VALID_DIR exists: {os.path.exists(VALID_DIR)}")
print(f"Files in TRAIN_DIR: {os.listdir(TRAIN_DIR)}")
print(f"Files in VALID_DIR: {os.listdir(VALID_DIR)}")
print(f"Number of images in TRAIN_DIR: {len(glob.glob(os.path.join(TRAIN_DIR, '*.jpg')) + glob.glob(os.path.join(TRAIN_DIR, '*.png')))}")
print(f"Number of images in VALID_DIR: {len(glob.glob(os.path.join(VALID_DIR, '*.jpg')) + glob.glob(os.path.join(VALID_DIR, '*.png')))}")

# Menyiapkan data loader untuk dataset pelatihan dan validasi
train_dataset = CustomDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
valid_dataset = CustomDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}")

print(f"Training directory: {TRAIN_DIR}")
print(f"Validation directory: {VALID_DIR}")

if len(train_dataset) == 0:
    raise ValueError(f"No training samples found in {TRAIN_DIR}")
if len(valid_dataset) == 0:
    raise ValueError(f"No validation samples found in {VALID_DIR}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)
