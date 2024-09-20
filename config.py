import torch

# Ukuran batch yang digunakan selama pelatihan
BATCH_SIZE = 4  # Meningkatkan ukuran batch untuk lebih banyak variasi per iterasi

# Ukuran gambar yang akan di-resize untuk pelatihan dan transformasi
RESIZE_TO = 800  # Meningkatkan ukuran untuk menangkap detail lebih baik

# Jumlah epoch yang akan dilatih
NUM_EPOCHS = 100  # Meningkatkan jumlah epoch untuk pelatihan yang lebih lama

# Learning rate
LEARNING_RATE = 0.0005  # Menambahkan learning rate yang lebih kecil

# Weight decay untuk regularisasi L2
WEIGHT_DECAY = 0.0005  # Menambahkan weight decay untuk regularisasi

# Tentukan apakah menggunakan GPU atau CPU
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Directory for training images and XML files
TRAIN_DIR = 'voc-fontawsome-dataset'

# Directory for validation images and XML files
VALID_DIR = 'voc-test-dataset'

import os
import xml.etree.ElementTree as ET

def get_classes(directory):
    classes = set(['background'])
    annotations_dir = os.path.join(directory, 'Annotations')
    for xml_file in os.listdir(annotations_dir):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(annotations_dir, xml_file))
            root = tree.getroot()
            for obj in root.findall('object'):
                classes.add(obj.find('name').text)
    return sorted(list(classes))

# Kelas yang digunakan, index 0 disediakan untuk background
CLASSES = get_classes(TRAIN_DIR)

# Jumlah kelas (termasuk background)
NUM_CLASSES = len(CLASSES)

print(f"Detected classes: {CLASSES}")
print(f"Number of classes (including background): {NUM_CLASSES}")

# Apakah akan menampilkan visualisasi gambar setelah transformasi pada data loader
VISUALIZE_TRANSFORMED_IMAGES = False

# Lokasi untuk menyimpan model dan grafik
OUT_DIR = './outputs'

# Menyimpan grafik loss setelah beberapa epoch
SAVE_PLOTS_EPOCH = 2  # simpan plot loss setelah setiap 2 epoch

# Menyimpan model setelah beberapa epoch
SAVE_MODEL_EPOCH = 2  # simpan model setelah setiap 2 epoch
