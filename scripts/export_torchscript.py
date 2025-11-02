import torch
import sys

from model_architecture import AgeGenderNet


def export_to_torchscript(input_model_path, output_model_path):
    print("="*60)
    print("EXPORTING MODEL TO TORCHSCRIPT")
    print("="*60)
    
    device = torch.device('cpu')
    
    print(f"\nLoading model from: {input_model_path}")
    model = AgeGenderNet()
    
    checkpoint = torch.load(input_model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded from checkpoint with state_dict")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded state_dict directly")
    
    model.eval()
    model.to(device)
    
    print("\nModel architecture:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nCreating example input...")
    example_input = torch.randn(1, 3, 224, 224, device=device)
    
    print("Testing forward pass...")
    with torch.no_grad():
        age, gender, emotion = model(example_input)
        print(f"  Age output shape: {age.shape}")
        print(f"  Gender output shape: {gender.shape}")
        print(f"  Emotion output shape: {emotion.shape}")
    
    print("\nTracing model with torch.jit.trace...")
    traced_model = torch.jit.trace(model, example_input)
    
    print("Verifying traced model...")
    with torch.no_grad():
        traced_age, traced_gender, traced_emotion = traced_model(example_input)
        
        age_diff = (age - traced_age).abs().max().item()
        gender_diff = (gender - traced_gender).abs().max().item()
        emotion_diff = (emotion - traced_emotion).abs().max().item()
        
        print(f"  Age diff: {age_diff:.6f}")
        print(f"  Gender diff: {gender_diff:.6f}")
        print(f"  Emotion diff: {emotion_diff:.6f}")
        
        if age_diff < 1e-5 and gender_diff < 1e-5 and emotion_diff < 1e-5:
            print("  ✓ Traced model matches original!")
        else:
            print("  ⚠ Warning: traced model has differences")
    
    print(f"\nSaving TorchScript model to: {output_model_path}")
    traced_model.save(output_model_path)
    
    import os
    size_mb = os.path.getsize(output_model_path) / (1024 * 1024)
    print(f"✓ TorchScript model saved: {size_mb:.2f} MB")
    
    print("\n" + "="*60)
    print("Testing reload...")
    reloaded = torch.jit.load(output_model_path, map_location=device)
    
    with torch.no_grad():
        reload_age, reload_gender, reload_emotion = reloaded(example_input)
        print(f"  Age: {reload_age.item():.2f}")
        print(f"  Gender: {reload_gender.item():.4f}")
        print(f"  Emotion: {reload_emotion[0].tolist()}")
    
    print("✓ TorchScript model works!")
    print("="*60)


if __name__ == '__main__':
    INPUT_MODEL = '../../models/agegenderemo.pt'
    OUTPUT_MODEL = '../../models/agegenderemo_traced.pt'
    
    export_to_torchscript(INPUT_MODEL, OUTPUT_MODEL)
    
    print("\n✓ Export complete!")
    print(f"\nUse this model in Rust: {OUTPUT_MODEL}")
