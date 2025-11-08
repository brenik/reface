# AgeGenderEmo Model Training Report

## Model Architecture
- **Model**: AgeGenderNet (Multi-task CNN)
- **Parameters**: 9,356,776 (35.73 MB)
- **Tasks**: Age regression, Gender classification, Emotion classification (8 classes)
- **Backbone**: Custom ResNet-style with CBAM attention

---

## Available Datasets
- **UTKFace**: 23,708 images (Age + Gender)
- **Emo96**: 30,627 images (Emotion)
- **FER2013**: 35,887 images (Emotion)
- **IMDB**: 460,724 images (Age + Gender)
- **Wiki**: 62,329 images (Age + Gender)
- **FFHQ**: 70,002 images (Age + Gender + Emotion)

---

## Training History

### Phase 1: Age & Gender Pre-training
- **Dataset**: UTKFace (23,708 images)
- **Tasks**: Age + Gender
- **Result**: Base model created

### Phase 2: Emotion Fine-tuning (Failed)
- **Dataset**: Emotion dataset (Emo96 or FER2013)
- **Approach**: Frozen backbone, emotion head only
- **Result**: Poor generalization

### Phase 3: Multi-dataset Training (Failed)
- **Datasets**: 
  - UTKFace for Age/Gender
  - Emotion dataset for Emotion
- **Approach**: Alternating batches
- **Epochs**: 5
- **Result**: 
  - Best val_loss: 3.8627
  - Training diverged

### Phase 4: Wiki Face Training (Failed)
- **Dataset**: Wiki (100 images test subset)
- **Epochs**: 5
- **Result**:
  - Best val_loss: 2.7590
  - Age MAE: 3.36 ± 4.04 years
  - Gender Acc: 94.00%

### Phase 5: Emotion-only Training (Poor)
- **Dataset**: Custom emotion dataset
- **Samples**: 252 train, 180 val
- **Epochs**: 20
- **Result**:
  - Best val_acc: 62.07%
  - Early stopping at epoch 20

### Phase 6: FFHQ Multi-task Training (Success)
- **Dataset**: FFHQ-Features (70,002 images with Azure annotations)
- **Samples**: 
  - Train: 58,331
  - Val: 9,735
- **Configuration**:
  - Batch size: 64
  - Learning rate: 0.0001
  - Optimizer: Adam (weight_decay=1e-5)
  - Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
  - Early stopping: patience=10
- **Loss weights**: Age=1.0, Gender=10.0, Emotion=5.0

---

## Training Progress (Phase 6, Epochs 16-30)

```
Epoch 16: val_loss=4.5026, Age MAE=2.52yr, Gender=97.19%, Emotion=93.35%
Epoch 17: val_loss=4.5273
Epoch 18: val_loss=4.8003
Epoch 19: val_loss=4.5593
Epoch 20: val_loss=4.7473
Epoch 21: val_loss=4.6198
Epoch 22: val_loss=4.7207
Epoch 23: val_loss=4.5623, LR reduced: 0.0001 → 0.00005
Epoch 24: val_loss=4.2871 ✓ BEST
         Age MAE=2.52yr, Gender=97.19%, Emotion=93.35%
Epoch 25: val_loss=4.3049
Epoch 26: val_loss=4.2583
Epoch 27: val_loss=4.1815
Epoch 28: val_loss=4.5362
Epoch 29: val_loss=4.3253
Epoch 30: val_loss=4.4531, LR reduced: 0.00005 → 0.000025
         Age MAE=2.54yr, Gender=99.35%, Emotion=93.85%
```

---

## Final Results

### Best Model (Epoch 24):
- **Validation Loss**: 4.2871
- **Age MAE**: 2.52 years
- **Gender Accuracy**: 97.19%
- **Emotion Accuracy**: 93.35%

### Hardware:
- **Device**: CPU
- **Training speed**: ~2.4 it/s (train), ~1.3 it/s (val)
- **Memory usage**: ~2080 MB

## Loss Functions

### Age Loss: MSE (Mean Squared Error)
```python
age_loss = mean((predicted_age - true_age)²) * 1.0
```
- Squares errors → penalizes large mistakes more
- Weight: 1.0

### Gender Loss: BCE (Binary Cross-Entropy)
```python
gender_loss = -[true*log(pred) + (1-true)*log(1-pred)] * 10.0
```
- Measures confidence of binary prediction (male/female)
- Weight: 10.0 (gender is easier, needs higher weight)

### Emotion Loss: KL Divergence
```python
emotion_loss = sum(true[i] * log(true[i]/pred[i])) * 5.0
```
- Compares probability distributions (soft labels)
- 8 emotions: anger, contempt, disgust, fear, happiness, neutral, sadness, surprise
- Weight: 5.0

### Total Loss
```python
total_loss = age_loss + gender_loss + emotion_loss
```

---

## Metrics (Evaluation)

### Age MAE (Mean Absolute Error)
```python
age_mae = mean(|predicted_age - true_age|)
```
- Average error in years
- Example: MAE=2.52 → average 2.52 years off

### Gender Accuracy
```python
accuracy = 100 * correct_predictions / total
```
- Percentage of correct male/female predictions

### Emotion Accuracy
```python
accuracy = 100 * (argmax(pred) == argmax(true)) / total
```
- Percentage where predicted emotion matches true emotion
- Uses argmax (highest probability) from 8 emotions

---

## Why Different Functions?

- **Age**: Regression → MSE
- **Gender**: Binary classification → BCE
- **Emotion**: Multi-class with soft labels → KL Divergence