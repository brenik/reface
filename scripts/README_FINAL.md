# 🚀 Тренування з Soft Labels - Фінальна версія

## 📁 Структура файлів:

```
/reface/scripts/06_reface_emo/
├── model_architecture.py      ← Модель (з Softmax в emotion_head)
├── train_ffhq_soft.py         ← Скрипт тренування
└── test_soft.py               ← Скрипт тестування
```

---

## 🔧 Що змінено в `model_architecture.py`:

### Було:
```python
self.emotion_head = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(64, 8)  # ← БЕЗ Softmax
)
```

### Стало:
```python
self.emotion_head = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(64, 8),
    nn.Softmax(dim=1)  # ← ДОДАНО Softmax
)
```

**Це єдина зміна в моделі!**

---

## 🎯 Що робить `train_ffhq_soft.py`:

1. **Завантажує** претреновану модель з `/reface/models/refacemo/agegenderemo.pt`
2. **Дотренує** на FFHQ датасеті з soft labels (KLDivLoss)
3. **Зберігає** оновлену модель назад в `agegenderemo.pt`

---

## ⚙️ Параметри тренування:

```python
PRETRAINED_MODEL = '/reface/models/refacemo/agegenderemo.pt'
IMAGES_DIR = '/reface/images/01_raw/ffhq-dataset/thumbnails'
JSON_DIR = '/reface/images/01_raw/ffhq-dataset/json'

BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001  # ← менший для fine-tuning
INPUT_SIZE = 224

age_weight = 1.0
gender_weight = 10.0
emotion_weight = 5.0
```

---

## 🚀 Команда для запуску:

```bash
cd /reface/
python3 scripts/06_reface_emo/train_ffhq_soft.py
```

---

## 📊 Soft Labels vs Hard Labels:

| Параметр | Hard Labels | Soft Labels |
|----------|-------------|-------------|
| **Target** | `[0,0,0,0,1,0,0,0]` | `[0.0,0.0,0.0,0.0,0.85,0.10,0.03,0.02]` |
| **Loss** | CrossEntropyLoss | KLDivLoss |
| **Output** | Logits | Probabilities (Softmax) |
| **Info** | 1 клас | Всі 8 ймовірностей |

---

## 💾 Збережені файли:

```
/reface/models/refacemo/
├── agegenderemo.pt              ← ОНОВЛЕНА модель (найкраща)
├── agegenderemo_best.pt         ← з метаданими
├── checkpoint_epoch_5.pt        ← чекпоінти
├── checkpoint_epoch_10.pt
└── ...

/reface/
├── training_history_epoch_5.png
├── training_history_epoch_10.png
└── training_history_final.png
```

---

## 📈 Очікувані результати:

- **Age MAE**: 4-6 років
- **Gender Accuracy**: 96-99%
- **Emotion Accuracy**: 65-75%

---

## 🔍 Формат виходу моделі:

```python
{
  "age": 25.3,
  "gender": "female",
  "max_emotion": "happiness",
  "emotion": {
    "anger": 0.02,
    "contempt": 0.01,
    "disgust": 0.01,
    "fear": 0.03,
    "happiness": 0.85,
    "neutral": 0.05,
    "sadness": 0.02,
    "surprise": 0.01
  }
}
```

---

## ✅ Чеклист перед запуском:

- [ ] Скопіюй оновлений `model_architecture.py` в `/reface/scripts/06_reface_emo/`
- [ ] Скопіюй `train_ffhq_soft.py` в `/reface/scripts/06_reface_emo/`
- [ ] Перевір що існує `/reface/models/refacemo/agegenderemo.pt`
- [ ] Перевір шляхи до IMAGES_DIR та JSON_DIR
- [ ] Запусти тренування!

---

Успішного тренування! 🚀
