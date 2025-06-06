import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, resnet50
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from custom_dataset import TwoPathImageDataset
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from waves.ldm import lr_scheduler

def print2file(logfile, *args):
    print(*args)
    print(file=logfile, *args)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train surrogate watermark clasifier.")
    parser.add_argument(
        "--image_size",
        type=int,
        choices=[256, 512],
        default=512,
        help="Image size",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="How many classes",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training (rest used for validation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--train_data_path_class0",
        type=str,
        required=True,
        help="Path to training images for class 0, un-watermarked should be class 0",
    )
    parser.add_argument(
        "--train_data_path_class1", 
        type=str,
        required=True,
        help="Path to training images for class 1, watermarked should be class 1",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=None,
        help="Limit the size of the training set. Use the full dataset if not specified.",
    )
    parser.add_argument(
        "--surrogate_model",
        type=str,
        default="ResNet18",
        help="Surrogate model.",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./models/surrogate_detectors",
    )
    parser.add_argument(
        "--model_save_name",
        type=str,
        default="resnet18",
    )
    # Training hyper-parameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=float,
        default=128,
        help="Batch size.",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
    )
    parsed_args = parser.parse_args()

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return parsed_args


def train_surrogate_classifier(args):
    # Data preprocessing/transformation
    transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ]
    )

    # Load datasets
    #full_train_dataset = ImageFolder(args.train_data_path, transform=transform)
    full_train_dataset = TwoPathImageDataset(
        path1=args.train_data_path_class0,
        path2=args.train_data_path_class1,
        transform=transform,
        train=True,
        train_ratio=args.train_ratio,
        seed=args.seed,
        args=args
    )


    if args.train_size is not None and 0 < args.train_size < len(full_train_dataset):
        indices = np.random.choice(
            len(full_train_dataset), args.train_size, replace=False
        )
        train_dataset = torch.utils.data.Subset(full_train_dataset, indices)
    else:
        train_dataset = full_train_dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    if args.do_eval:
        valid_dataset = TwoPathImageDataset(
            path1=args.train_data_path_class0,
            path2=args.train_data_path_class1,
            transform=transform,
            train=False,
            train_ratio=args.train_ratio,
            seed=args.seed,
            args=args
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    print2file(args.log_file, f"Training on {len(train_dataset)} samples.")
    print2file(args.log_file, f"Validation on {len(valid_dataset)} samples.")

    # Load pretrained ResNet18 and modify the final layer
    if args.surrogate_model == "ResNet18":
        model = resnet18(pretrained=True)
    elif args.surrogate_model == "ResNet50":
        model = resnet50(pretrained=True)
    else:
        raise ValueError(f"Unknown model: {args.surrogate_model}")

    if model.fc.out_features != args.num_classes:
        # Modify the final layer only if the number of output features doesn't match
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)

    model = model.to(args.device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    #lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    best_val_accuracy = 0.0
    best_model_state = None
    train_accs = []
    val_accs = []
    losses = []
    lrs = []

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            total_loss += loss.item()

            # Calculate predictions for training accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        #lr_scheduler.step()

        #lrs.append(lr_scheduler.get_last_lr()[0])
        train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        print2file(args.log_file, 
            f"Epoch [{epoch + 1}/{args.num_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%"
        )
        train_accs.append(train_accuracy)
        losses.append(train_loss)

        if args.do_eval:
            # Evaluation on the validation set
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(args.device), labels.to(args.device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total
            print2file(args.log_file, f"Validation Accuracy: {val_accuracy:.2f}%")

            # Update best validation accuracy and model state
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict().copy()  # Copy the model state
                print2file(args.log_file, 
                    f"New best model found at epoch {epoch + 1} with validation accuracy: {val_accuracy:.2f}%"
                )
            val_accs.append(val_accuracy)

    print2file(args.log_file, "Training complete!")

    # Save the entire model
    # Save the best model based on validation accuracy
    if best_model_state is not None:
        save_path_best = os.path.join(
            args.model_save_path, args.model_save_name + "_acc" + str(best_val_accuracy) + ".pth"
        )
        torch.save(best_model_state, save_path_best)
        print2file(args.log_file, 
            f"Best model saved to {save_path_best} with validation accuracy: {best_val_accuracy:.2f}%"
        )
    else:
        save_path_full = os.path.join(
            args.model_save_path, args.model_save_name + ".pth"
        )
        torch.save(model.state_dict(), save_path_full)
        print2file(args.log_file, f"Entire model saved to {save_path_full}")
    return train_accs, val_accs, losses, lrs, save_path_best


if __name__ == "__main__":
    args = parse_arguments()
    print2file(args.log_file, args)

    os.makedirs(args.model_save_path, exist_ok=True)
    train_surrogate_classifier(args)
