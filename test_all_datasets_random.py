import numpy as np
import pickle
import os
import kagglehub
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import urllib.request
import tarfile
from PIL import Image
import glob
import csv


def download_datasets():
    """Download Weather and Skin Cancer datasets using kagglehub"""
    print("Downloading Weather dataset...")
    weather_path = kagglehub.dataset_download("pratik2901/multiclass-weather-dataset")
    print(f"Weather dataset downloaded to: {weather_path}")
    
    print("\nDownloading Skin Cancer dataset...")
    skin_cancer_path = kagglehub.dataset_download("fatemehmehrparvar/skin-cancer-detection")
    print(f"Skin Cancer dataset downloaded to: {skin_cancer_path}")
    
    return weather_path, skin_cancer_path


def load_cifar10_test(data_dir=None):
    """Load CIFAR-10 test dataset"""
    if data_dir is None:
        data_dir = os.path.join(os.path.expanduser("~"), ".cache", "cifar-10-batches-py")
    
    # Download CIFAR-10 if not exists
    if not os.path.exists(data_dir):
        print("Downloading CIFAR-10...")
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        tar_path = os.path.join(os.path.dirname(data_dir), "cifar-10-python.tar.gz")
        
        os.makedirs(os.path.dirname(data_dir), exist_ok=True)
        urllib.request.urlretrieve(url, tar_path)
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=os.path.dirname(data_dir))
        
        os.remove(tar_path)
    
    # Load test batch
    test_file = os.path.join(data_dir, 'test_batch')
    with open(test_file, 'rb') as f:
        batch_dict = pickle.load(f, encoding='bytes')
        test_data = batch_dict[b'data']
        test_labels = batch_dict[b'labels']
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    return test_data, test_labels, class_names, 10


def load_ood_cv_test(data_dir=None):
    """Load OOD-CV dataset from Zhao et al. 2022 ECCV"""
    if data_dir is None:
        # Use the actual OOD-CV-Cls directory in the current project
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OOD-CV-Cls")
    
    print("OOD-CV15 dataset information:")
    print("- 10 object categories from PASCAL3D+ with OOD variations")
    print("- Test set: phase-1 + phase-2 images")
    print("- Variations: Pose, Shape, Texture, Context, Weather")
    print("- Tasks: Image classification, Object detection, 3D pose estimation")
    print(f"- Dataset path: {data_dir}")
    
    test_data = []
    test_labels = []
    all_images = []
    
    # Define the class names mapping
    class_mapping = {
        'aeroplane': 0, 'bicycle': 1, 'boat': 2, 'bottle': 3, 'bus': 4,
        'car': 5, 'chair': 6, 'diningtable': 7, 'motorbike': 8, 'sofa': 9, 'train': 10
    }
    
    # Get unique classes that exist in the dataset
    unique_classes = set()
    
    # Load phase-1 data
    phase1_dir = os.path.join(data_dir, "phase-1")
    phase1_images_dir = os.path.join(phase1_dir, "images")
    phase1_labels_file = os.path.join(phase1_dir, "labels.csv")
    
    if os.path.exists(phase1_labels_file) and os.path.exists(phase1_images_dir):
        print("Loading phase-1 data...")
        import csv
        with open(phase1_labels_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row['imgs']
                label = row['labels']
                img_path = os.path.join(phase1_images_dir, img_name)
                
                if os.path.exists(img_path) and label in class_mapping:
                    all_images.append((img_path, class_mapping[label], label))
                    unique_classes.add(label)
    
    # Load phase-2 data
    phase2_dir = os.path.join(data_dir, "phase-2")
    phase2_images_dir = os.path.join(phase2_dir, "images")
    phase2_labels_file = os.path.join(phase2_dir, "labels.csv")
    
    if os.path.exists(phase2_labels_file) and os.path.exists(phase2_images_dir):
        print("Loading phase-2 data...")
        import csv
        with open(phase2_labels_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row['imgs']
                label = row['labels']
                img_path = os.path.join(phase2_images_dir, img_name)
                
                if os.path.exists(img_path) and label in class_mapping:
                    all_images.append((img_path, class_mapping[label], label))
                    unique_classes.add(label)
    
    # Create sorted class names list based on what's actually in the dataset
    class_names_sorted = sorted(unique_classes)
    class_names = class_names_sorted
    
    # Re-map labels to sequential indices
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    print(f"Found {len(all_images)} images")
    print(f"Classes: {class_names}")
    
    # Load all images
    for img_path, _, label_name in all_images:
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((64, 64))  # Resize to standard size
            img_array = np.array(img)  # Shape: (64, 64, 3)
            # Convert to CIFAR-10 format: transpose from (H, W, C) to (C, H, W) then flatten
            img_array = img_array.transpose(2, 0, 1).flatten()  # Now in (C, H, W) format flattened
            test_data.append(img_array)
            test_labels.append(class_to_idx[label_name])
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
    
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    
    print(f"Successfully loaded {len(test_labels)} test images")
    
    return test_data, test_labels, class_names, len(class_names)


def load_weather_dataset(data_path):
    """Load Multi-class Weather Dataset"""
    print(f"Loading Weather dataset from: {data_path}")
    
    # Find the dataset directory - it's in "Multi-class Weather Dataset" folder
    dataset_dir = os.path.join(data_path, "Multi-class Weather Dataset")
    if not os.path.exists(dataset_dir):
        dataset_dir = data_path
    
    # Weather classes as specified
    class_names = ['Sunrise', 'Shine', 'Rain', 'Cloudy']
    test_data = []
    test_labels = []
    all_images = []
    
    # Collect all images from each class
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_path):
            # Load images from this class
            image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            image_files = []
            for pattern in image_patterns:
                image_files.extend(glob.glob(os.path.join(class_path, pattern)))
            
            # Store with class label
            for img_path in image_files:
                all_images.append((img_path, class_idx))
    
    # Use all images in the dataset
    test_images = all_images
    
    print(f"Total images found: {len(all_images)}")
    print(f"Using all {len(test_images)} images for evaluation")
    
    # Load the test images
    for img_path, class_idx in test_images:
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((64, 64))  # Resize to standard size
            img_array = np.array(img)  # Shape: (64, 64, 3)
            # Convert to CIFAR-10 format: transpose from (H, W, C) to (C, H, W) then flatten
            img_array = img_array.transpose(2, 0, 1).flatten()  # Now in (C, H, W) format flattened
            test_data.append(img_array)
            test_labels.append(class_idx)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
    
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    
    print(f"Successfully loaded {len(test_labels)} test images")
    print(f"Classes: {class_names}")
    
    return test_data, test_labels, class_names, len(class_names)


def load_skin_cancer_dataset(data_path):
    """Load Skin Cancer dataset for melanoma detection"""
    print(f"Loading Skin Cancer dataset from: {data_path}")
    print("Dataset: University of Waterloo skin cancer dataset")
    print("Sources: DermIS and DermQuest databases")
    print("Task: Binary classification (healthy/melanoma)")
    
    test_data = []
    test_labels = []
    class_names = ['healthy', 'melanoma']  # Binary classification
    
    # Navigate to the actual data directory
    skin_data_paths = [
        os.path.join(data_path, "skin_image_data_set-1", "Skin Image Data Set-1", "skin_data"),
        os.path.join(data_path, "skin_image_data_set-2", "Skin Image Data Set-2", "skin_data")
    ]
    
    found_data = False
    melanoma_images = []
    healthy_images = []
    
    for base_path in skin_data_paths:
        if os.path.exists(base_path):
            # Load melanoma (melanoma) images
            melanoma_paths = [
                os.path.join(base_path, "melanoma", "dermIS"),
                os.path.join(base_path, "melanoma", "dermquest")
            ]
            
            for mel_path in melanoma_paths:
                if os.path.exists(mel_path):
                    found_data = True
                    # Load only original images (not contour images)
                    image_files = glob.glob(os.path.join(mel_path, "*_orig.jpg"))
                    melanoma_images.extend(image_files)
            
            # Load notmelanoma (healthy) images
            notmel_path = os.path.join(base_path, "notmelanoma")
            if os.path.exists(notmel_path):
                found_data = True
                # Look for subdirectories
                for subdir in os.listdir(notmel_path):
                    subdir_path = os.path.join(notmel_path, subdir)
                    if os.path.isdir(subdir_path):
                        image_files = glob.glob(os.path.join(subdir_path, "*_orig.jpg"))
                        healthy_images.extend(image_files)
    
    if found_data:
        print(f"Found {len(melanoma_images)} cancerous images")
        print(f"Found {len(healthy_images)} healthy images")
        
        # Load balanced dataset
        min_samples = min(len(melanoma_images), len(healthy_images))
        
        # Load cancerous images
        for img_path in melanoma_images[:min_samples]:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((128, 128))  # Center cropped as mentioned
                img_array = np.array(img)  # Shape: (128, 128, 3)
                # Convert to CIFAR-10 format: transpose from (H, W, C) to (C, H, W) then flatten
                img_array = img_array.transpose(2, 0, 1).flatten()  # Now in (C, H, W) format flattened
                test_data.append(img_array)
                test_labels.append(1)  # cancerous = 1
            except Exception as e:
                continue
        
        # Load healthy images
        for img_path in healthy_images[:min_samples]:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((128, 128))
                img_array = np.array(img)  # Shape: (128, 128, 3)
                # Convert to CIFAR-10 format: transpose from (H, W, C) to (C, H, W) then flatten
                img_array = img_array.transpose(2, 0, 1).flatten()  # Now in (C, H, W) format flattened
                test_data.append(img_array)
                test_labels.append(0)  # healthy = 0
            except Exception as e:
                continue
    else:
        raise ValueError("Could not find skin cancer data")
    
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    
    print(f"Classes: {class_names}")
    print(f"Total test images: {len(test_labels)}")
    
    return test_data, test_labels, class_names, len(class_names)


def random_classifier(num_samples, num_classes):
    """Generate random predictions"""
    return np.random.randint(0, num_classes, size=num_samples)


def evaluate_performance(y_true, y_pred, class_names, dataset_name):
    """Evaluate classification performance"""
    print(f"\n{'='*60}")
    print(f"Results for {dataset_name}")
    print(f"{'='*60}\n")
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Precision, Recall, F1-score
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    print(f"\nPer-class metrics:")
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 60)
    for i in range(len(class_names)):
        if i < len(support_per_class):
            print(f"{class_names[i]:<20} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} "
                  f"{f1_per_class[i]:<10.4f} {support_per_class[i]:<10}")
    
    # Expected accuracy for random classifier
    expected_accuracy = 1.0 / len(class_names)
    print(f"\nExpected accuracy for random classifier: {expected_accuracy:.4f}")
    print(f"Difference from expected: {accuracy - expected_accuracy:.4f}")
    
    # Confusion matrix statistics
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nTotal test samples: {len(y_true)}")
    print(f"Correctly classified: {np.trace(cm)}")
    print(f"Misclassified: {len(y_true) - np.trace(cm)}")


def main():
    # Download Weather and Skin Cancer datasets
    weather_path, skin_cancer_path = download_datasets()
    
    # Dataset configurations
    datasets = [
        {
            'name': 'CIFAR-10',
            'loader': load_cifar10_test,
            'args': []
        },
        {
            'name': 'OOD-CV15',
            'loader': load_ood_cv_test,
            'args': []
        },
        {
            'name': 'Weather Classification',
            'loader': load_weather_dataset,
            'args': [weather_path]
        },
        {
            'name': 'Skin Cancer',
            'loader': load_skin_cancer_dataset,
            'args': [skin_cancer_path]
        }
    ]
    
    # Test each dataset
    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"Loading {dataset['name']} dataset...")
        
        try:
            # Load dataset
            test_data, test_labels, class_names, num_classes = dataset['loader'](*dataset['args'])
            print(f"Loaded {len(test_labels)} test images from {num_classes} classes")
            
            # Skip if no data loaded
            if len(test_labels) == 0:
                print(f"No data loaded for {dataset['name']}, skipping...")
                continue
            
            # Random predictions
            predictions = random_classifier(len(test_labels), num_classes)
            
            # Evaluate
            evaluate_performance(test_labels, predictions, class_names, dataset['name'])
            
        except Exception as e:
            print(f"Error loading {dataset['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()