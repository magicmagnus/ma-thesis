import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple

def print2file(logfile, *args):
    print(*args)
    print(file=logfile, *args)

class TwoPathImageDataset(Dataset):
    def __init__(
        self, 
        path1: str,
        path2: str,
        transform: Optional[Callable] = None,
        train: bool = True,
        train_ratio: float = 0.8,
        seed: int = 42,
        label1: int = 0,
        label2: int = 1,
        args=None
    ):
        self.transform = transform
        np.random.seed(seed)

        print2file(args.log_file, ("\nTraining Set:\n") if train else "\nTest Set:\n")
        
        # Get all image files from both paths
        self.images = []
        self.labels = []
        
        # Load images from both paths
        for path, label in [(path1, label1), (path2, label2)]:
            img_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Create train/test split for this class
            n_samples = len(img_files)
            n_train = int(n_samples * train_ratio)
            indices = np.random.permutation(n_samples)
            
            # Select appropriate indices based on train/test mode
            if train:
                selected_indices = indices[:n_train]
            else:
                selected_indices = indices[n_train:]
                
            
            # Add selected images to dataset
            for idx in selected_indices:
                img_name = img_files[idx]
                self.images.append(os.path.join(path, img_name))
                self.labels.append(label)

            print2file(args.log_file, f'Loaded {len(selected_indices)} images with label {label} from \n{path} \n')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label