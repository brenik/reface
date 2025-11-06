import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from model_architecture import AgeGenderNet


EMOTION_LABELS = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']


class FFHQDataset(Dataset):
    def __init__(self, images_dir, json_dir, transform=None, split='train'):
        self.images_dir = images_dir
        self.json_dir = json_dir
        self.transform = transform
        self.split = split
        
        self.samples = []
        self._load_data()
    
    def _load_data(self):
        print(f"Loading {self.split} data...")
        
        json_files = sorted([f for f in os.listdir(self.json_dir) if f.endswith('.json')])
        
        if self.split == 'train':
            json_files = json_files[:60000]
        else:
            json_files = json_files[60000:]
        
        for json_file in tqdm(json_files, desc=f"Loading {self.split}"):
            img_name = json_file.replace('.json', '.png')
            img_path = os.path.join(self.images_dir, img_name)
            json_path = os.path.join(self.json_dir, json_file)
            
            if not os.path.exists(img_path):
                continue
            
            if not os.path.exists(json_path):
                continue
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    if len(data) == 0:
                        continue
                    data = data[0]
                
                if not isinstance(data, dict):
                    continue
                
                face_attrs = data.get('faceAttributes', {})
                if not face_attrs:
                    continue
                
                age = face_attrs.get('age', None)
                if age is None or age == 0.0:
                    continue
                
                gender = face_attrs.get('gender', None)
                if gender is None:
                    continue
                gender_binary = 1 if gender.lower() == 'male' else 0
                
                emotion_dict = face_attrs.get('emotion', {})
                if not emotion_dict:
                    continue
                
                emotion_probs = [emotion_dict.get(label, 0.0) for label in EMOTION_LABELS]
                
                total_prob = sum(emotion_probs)
                if total_prob == 0:
                    continue
                emotion_probs = [p / total_prob for p in emotion_probs]
                
                emotion_class = np.argmax(emotion_probs)
                
                self.samples.append({
                    'img_path': img_path,
                    'age': float(age),
                    'gender': gender_binary,
                    'emotion_probs': emotion_probs,
                    'emotion_class': emotion_class
                })
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} samples for {self.split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = Image.open(sample['img_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        age = torch.tensor(sample['age'], dtype=torch.float32)
        gender = torch.tensor(sample['gender'], dtype=torch.float32)
        emotion_probs = torch.tensor(sample['emotion_probs'], dtype=torch.float32)
        emotion_class = torch.tensor(sample['emotion_class'], dtype=torch.long)
        
        return image, age, gender, emotion_probs, emotion_class


def get_transforms(input_size=224, augment=True):
    if augment:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


class MultiTaskLoss(nn.Module):
    def __init__(self, age_weight=1.0, gender_weight=1.0, emotion_weight=1.0):
        super(MultiTaskLoss, self).__init__()
        self.age_weight = age_weight
        self.gender_weight = gender_weight
        self.emotion_weight = emotion_weight
        
        self.age_loss = nn.L1Loss()
        self.gender_loss = nn.BCELoss()
        self.emotion_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, age_pred, gender_pred, emotion_pred, age_true, gender_true, emotion_true):
        loss_age = self.age_loss(age_pred.squeeze(), age_true)
        loss_gender = self.gender_loss(gender_pred.squeeze(), gender_true)
        
        emotion_pred_log = torch.log(emotion_pred + 1e-10)
        loss_emotion = self.emotion_loss(emotion_pred_log, emotion_true)
        
        total_loss = (self.age_weight * loss_age + 
                      self.gender_weight * loss_gender + 
                      self.emotion_weight * loss_emotion)
        
        return total_loss, loss_age, loss_gender, loss_emotion


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    
    running_loss = 0.0
    running_age_loss = 0.0
    running_gender_loss = 0.0
    running_emotion_loss = 0.0
    
    age_mae = 0.0
    gender_correct = 0
    emotion_correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for images, ages, genders, emotion_probs, emotion_classes in pbar:
        images = images.to(device)
        ages = ages.to(device)
        genders = genders.to(device)
        emotion_probs = emotion_probs.to(device)
        emotion_classes = emotion_classes.to(device)
        
        optimizer.zero_grad()
        
        age_pred, gender_pred, emotion_pred = model(images)
        
        loss, loss_age, loss_gender, loss_emotion = criterion(
            age_pred, gender_pred, emotion_pred,
            ages, genders, emotion_probs
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        running_age_loss += loss_age.item()
        running_gender_loss += loss_gender.item()
        running_emotion_loss += loss_emotion.item()
        
        age_mae += torch.abs(age_pred.squeeze() - ages).sum().item()
        gender_correct += ((gender_pred.squeeze() > 0.5) == genders).sum().item()
        emotion_correct += (emotion_pred.argmax(dim=1) == emotion_classes).sum().item()
        total += images.size(0)
        
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'age_mae': f'{age_mae / total:.2f}',
            'gender_acc': f'{100. * gender_correct / total:.2f}%',
            'emotion_acc': f'{100. * emotion_correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_age_loss = running_age_loss / len(dataloader)
    epoch_gender_loss = running_gender_loss / len(dataloader)
    epoch_emotion_loss = running_emotion_loss / len(dataloader)
    
    epoch_age_mae = age_mae / total
    epoch_gender_acc = 100. * gender_correct / total
    epoch_emotion_acc = 100. * emotion_correct / total
    
    return (epoch_loss, epoch_age_loss, epoch_gender_loss, epoch_emotion_loss,
            epoch_age_mae, epoch_gender_acc, epoch_emotion_acc)


def validate(model, dataloader, criterion, device):
    model.eval()
    
    running_loss = 0.0
    running_age_loss = 0.0
    running_gender_loss = 0.0
    running_emotion_loss = 0.0
    
    age_mae = 0.0
    gender_correct = 0
    emotion_correct = 0
    total = 0
    
    with torch.no_grad():
        for images, ages, genders, emotion_probs, emotion_classes in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            ages = ages.to(device)
            genders = genders.to(device)
            emotion_probs = emotion_probs.to(device)
            emotion_classes = emotion_classes.to(device)
            
            age_pred, gender_pred, emotion_pred = model(images)
            
            loss, loss_age, loss_gender, loss_emotion = criterion(
                age_pred, gender_pred, emotion_pred,
                ages, genders, emotion_probs
            )
            
            running_loss += loss.item()
            running_age_loss += loss_age.item()
            running_gender_loss += loss_gender.item()
            running_emotion_loss += loss_emotion.item()
            
            age_mae += torch.abs(age_pred.squeeze() - ages).sum().item()
            gender_correct += ((gender_pred.squeeze() > 0.5) == genders).sum().item()
            emotion_correct += (emotion_pred.argmax(dim=1) == emotion_classes).sum().item()
            total += images.size(0)
    
    val_loss = running_loss / len(dataloader)
    val_age_loss = running_age_loss / len(dataloader)
    val_gender_loss = running_gender_loss / len(dataloader)
    val_emotion_loss = running_emotion_loss / len(dataloader)
    
    val_age_mae = age_mae / total
    val_gender_acc = 100. * gender_correct / total
    val_emotion_acc = 100. * emotion_correct / total
    
    return (val_loss, val_age_loss, val_gender_loss, val_emotion_loss,
            val_age_mae, val_gender_acc, val_emotion_acc)


def plot_training_history(history, save_path='training_history.png'):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['train_age_mae'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_age_mae'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_title('Age MAE', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (years)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(epochs, history['train_gender_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 2].plot(epochs, history['val_gender_acc'], 'r-', label='Val', linewidth=2)
    axes[0, 2].set_title('Gender Accuracy', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy (%)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, history['train_emotion_acc'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_emotion_acc'], 'r-', label='Val', linewidth=2)
    axes[1, 0].set_title('Emotion Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, history['train_age_loss'], 'b-', label='Age', linewidth=2)
    axes[1, 1].plot(epochs, history['train_gender_loss'], 'g-', label='Gender', linewidth=2)
    axes[1, 1].plot(epochs, history['train_emotion_loss'], 'orange', label='Emotion', linewidth=2)
    axes[1, 1].set_title('Training Task Losses', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(epochs, history['val_age_loss'], 'b-', label='Age', linewidth=2)
    axes[1, 2].plot(epochs, history['val_gender_loss'], 'g-', label='Gender', linewidth=2)
    axes[1, 2].plot(epochs, history['val_emotion_loss'], 'orange', label='Emotion', linewidth=2)
    axes[1, 2].set_title('Validation Task Losses', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history saved to {save_path}")


def main():
    PRETRAINED_MODEL = '../../models/agegenderemo.pt'
    IMAGES_DIR = '../../images/thumbnails'
    JSON_DIR = '../../images/json'

    CHECKPOINTS_DIR = '../../models/checkpoints'
    RESUME_CHECKPOINT_NAME = 'agegenderemo_best.pt'
    
    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.0001
    INPUT_SIZE = 224
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    train_transform = get_transforms(INPUT_SIZE, augment=True)
    val_transform = get_transforms(INPUT_SIZE, augment=False)
    
    print("\n" + "="*60)
    train_dataset = FFHQDataset(IMAGES_DIR, JSON_DIR, transform=train_transform, split='train')
    val_dataset = FFHQDataset(IMAGES_DIR, JSON_DIR, transform=val_transform, split='val')
    print("="*60 + "\n")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=torch.cuda.is_available())
    
    model = AgeGenderNet().to(device)
    
    print("="*60)
    print("Loading pretrained weights from:")
    print(PRETRAINED_MODEL)
    print("="*60)
    
    try:
        checkpoint = torch.load(PRETRAINED_MODEL, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        print("✓ Pretrained weights loaded successfully!")
        print("✓ Continuing training with soft labels")
        
    except Exception as e:
        print(f"Warning: Could not load pretrained weights: {e}")
        print("Starting training from scratch...")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")
    print(f"Estimated size: {total_params * 4 / (1024**2):.2f} MB\n")
    
    criterion = MultiTaskLoss(age_weight=1.0, gender_weight=10.0, emotion_weight=5.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                      patience=5)
    
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    RESUME_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, RESUME_CHECKPOINT_NAME)
    start_epoch = 1
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_age_loss': [], 'val_age_loss': [],
        'train_gender_loss': [], 'val_gender_loss': [],
        'train_emotion_loss': [], 'val_emotion_loss': [],
        'train_age_mae': [], 'val_age_mae': [],
        'train_gender_acc': [], 'val_gender_acc': [],
        'train_emotion_acc': [], 'val_emotion_acc': []
    }
    
    if os.path.exists(RESUME_CHECKPOINT):
        print("="*60)
        print("Found checkpoint! Resuming training...")
        print(f"Loading: {RESUME_CHECKPOINT}")
        print("="*60)
        try:
            checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            patience_counter = checkpoint.get('patience_counter', 0)
            history = checkpoint.get('history', history)
            
            print(f"✓ Resuming from epoch {start_epoch}")
            print(f"✓ Best val loss so far: {best_val_loss:.4f}")
            print(f"✓ Patience counter: {patience_counter}")
            print("="*60 + "\n")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Starting from beginning...")
            start_epoch = 1
    else:
        print("="*60)
        print("No checkpoint found. Starting from scratch.")
        print(f"Looking for: {RESUME_CHECKPOINT}")
        print("="*60 + "\n")
    
    early_stop_patience = 10
    
    print("="*60)
    print("Starting FINE-TUNING with SOFT LABELS")
    print(f"Epochs: {start_epoch} → {NUM_EPOCHS}")
    print(f"Early stopping patience: {early_stop_patience} epochs")
    print("="*60 + "\n")
    
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, criterion, device)
        
        (train_loss, train_age_loss, train_gender_loss, train_emotion_loss,
         train_age_mae, train_gender_acc, train_emotion_acc) = train_metrics
        
        (val_loss, val_age_loss, val_gender_loss, val_emotion_loss,
         val_age_mae, val_gender_acc, val_emotion_acc) = val_metrics
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_age_loss'].append(train_age_loss)
        history['val_age_loss'].append(val_age_loss)
        history['train_gender_loss'].append(train_gender_loss)
        history['val_gender_loss'].append(val_gender_loss)
        history['train_emotion_loss'].append(train_emotion_loss)
        history['val_emotion_loss'].append(val_emotion_loss)
        history['train_age_mae'].append(train_age_mae)
        history['val_age_mae'].append(val_age_mae)
        history['train_gender_acc'].append(train_gender_acc)
        history['val_gender_acc'].append(val_gender_acc)
        history['train_emotion_acc'].append(train_emotion_acc)
        history['val_emotion_acc'].append(val_emotion_acc)
        
        print(f"\nTrain - Loss: {train_loss:.4f} | Age MAE: {train_age_mae:.2f} | "
              f"Gender Acc: {train_gender_acc:.2f}% | Emotion Acc: {train_emotion_acc:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f} | Age MAE: {val_age_mae:.2f} | "
              f"Gender Acc: {val_gender_acc:.2f}% | Emotion Acc: {val_emotion_acc:.2f}%")
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            print(f"✓ Improvement! (val_loss: {val_loss:.4f})")
            
            torch.save(model.state_dict(), PRETRAINED_MODEL)
            print(f"  ✓ Saved: agegenderemo.pt")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_age_mae': val_age_mae,
                'val_gender_acc': val_gender_acc,
                'val_emotion_acc': val_emotion_acc,
                'history': history
            }, os.path.join(CHECKPOINTS_DIR, f'checkpoint_epoch_{epoch}.pt'))
            print(f"  ✓ Saved: checkpoints/checkpoint_epoch_{epoch}.pt")
            
            if epoch % 5 == 0:
                plot_training_history(history, f'training_history_epoch_{epoch}.png')
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'history': history
        }, RESUME_CHECKPOINT)
        print(f"  ✓ Saved: checkpoints/{RESUME_CHECKPOINT_NAME} (for resume)")
        
        if patience_counter >= early_stop_patience:
            print("\n" + "="*60)
            print(f"Early stopping triggered! No improvement for {early_stop_patience} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print("="*60)
            break
    
    if patience_counter >= early_stop_patience:
        plot_training_history(history, 'training_history_early_stop.png')
        print(f"\nTraining stopped early at epoch {epoch}")
        print(f"Final model saved with best val_loss: {best_val_loss:.4f}")
    else:
        plot_training_history(history, 'training_history_final.png')
        print("\n" + "="*60)
        print("Training completed all epochs!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Model saved to: {PRETRAINED_MODEL}")
        print("="*60)


if __name__ == '__main__':
    main()
