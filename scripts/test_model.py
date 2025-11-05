import torch
import json
from PIL import Image
from torchvision import transforms
from model_architecture import AgeGenderNet


EMOTION_LABELS = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']


def predict_with_labels(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(device)
        age, gender, emotion_probs = model(image_tensor)

        age_value = age.item()
        gender_value = "male" if gender.item() > 0.5 else "female"

        emotion_probs_np = emotion_probs.cpu().numpy()[0]
        max_emotion_idx = emotion_probs_np.argmax()
        max_emotion = EMOTION_LABELS[max_emotion_idx]

        emotion_dict = {label: float(prob) for label, prob in zip(EMOTION_LABELS, emotion_probs_np)}

        result = {
            "age": round(age_value, 1),
            "gender": gender_value,
            "max_emotion": max_emotion,
            "emotion": emotion_dict
        }

        return result


def load_model(model_path, device):
    model = AgeGenderNet().to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
            if 'epoch' in checkpoint:
                print(f"Epoch: {checkpoint['epoch']}")
            if 'val_loss' in checkpoint:
                print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
            if 'val_age_mae' in checkpoint:
                print(f"Age MAE: {checkpoint['val_age_mae']:.2f} years")
            if 'val_gender_acc' in checkpoint:
                print(f"Gender Accuracy: {checkpoint['val_gender_acc']:.2f}%")
            if 'val_emotion_acc' in checkpoint:
                print(f"Emotion Accuracy: {checkpoint['val_emotion_acc']:.2f}%")
        else:
            model.load_state_dict(checkpoint)
            print(f"Model loaded from {model_path}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Model loaded from {model_path}")

    model.eval()
    return model


def preprocess_image(image_path, input_size=224):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)

    return image_tensor


def test_single_image(model_path, image_path, device='cuda'):
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")

    device = torch.device(device)

    model = load_model(model_path, device)

    image_tensor = preprocess_image(image_path)

    result = predict_with_labels(model, image_tensor, device)

    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\nAge: {result['age']} years")
    print(f"Gender: {result['gender']}")
    print(f"Max Emotion: {result['max_emotion']}")
    print(f"\nEmotion Probabilities:")
    for emotion, prob in sorted(result['emotion'].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 50)
        print(f"  {emotion:12s}: {prob:.4f} {bar}")
    print("="*60)

    return result


def test_multiple_images(model_path, image_paths, device='cuda'):
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")

    device = torch.device(device)

    model = load_model(model_path, device)

    results = []

    for image_path in image_paths:
        print(f"\nProcessing: {image_path}")
        image_tensor = preprocess_image(image_path)
        result = predict_with_labels(model, image_tensor, device)
        results.append({
            'image_path': image_path,
            'result': result
        })

        print(f"Age: {result['age']}, Gender: {result['gender']}, "
              f"Emotion: {result['max_emotion']} ({result['emotion'][result['max_emotion']]:.2f})")

    return results


def save_results_to_json(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    MODEL_PATH = '../../models/agegenderemo.pt'
    IMAGE_PATH = '../../images/00000.png'

    result = test_single_image(MODEL_PATH, IMAGE_PATH, device='cuda')

    save_results_to_json([{'image': IMAGE_PATH, 'prediction': result}],
                         'inference_result.json')