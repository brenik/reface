# Reface: Age, Gender & Emotion Recognition Server

High-performance Rust inference server for facial attribute prediction using PyTorch-trained CNN model.

## Overview

Rust web server (Actix) serving a deep learning model for real-time facial analysis:
- **Age estimation** (MAE: ~2 years)
- **Gender classification** (98% accuracy)
- **Emotion recognition** with soft labels (94% accuracy)

## Architecture

### Model
- Custom ResNet + CBAM attention (9.4M parameters)
- Trained in PyTorch on FFHQ dataset (70k faces)
- Exported to TorchScript for Rust inference
- 8 emotions: anger, contempt, disgust, fear, happiness, neutral, sadness, surprise

### Server
- **Framework**: Actix-web (async Rust)
- **Inference**: tch-rs (PyTorch C++ bindings)
- **Speed**: ~19ms per image on CPU
- **Payload**: Up to 10MB

## Project Structure

```
rust_reface/
├── http/
│   └── api_examples.http        # HTTP client tests
├── images/
│   ├── 00000.png                # Sample images for testing
│   ├── 00001.png
│   └── ...
├── models/
│   └── agegenderemo_traced.pt   # TorchScript model (ready for Rust)
├── scripts/                      # Python (model training/export only)
│   ├── export_torchscript.py    # Convert .pt → TorchScript
│   ├── model_architecture.py    # PyTorch model definition
│   ├── train_ffhq_soft.py       # Training script
│   └── test_soft.py             # Python testing
├── src/
│   ├── main.rs                  # API server
│   └── model.rs                 # Inference engine
├── Cargo.toml
└── README.md
```

## Quick Start

### 1. Prerequisites

Install Rust and libtorch:

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# libtorch (CPU)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch-*.zip
export LIBTORCH=$PWD/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
```

### 2. Build & Run

```bash
cargo build --release
cargo run --release
```

Server starts on `http://0.0.0.0:8080`

### 3. Testing with HTTP Client

Use the provided `.http` file in your IDE (IntelliJ IDEA, VS Code with REST Client extension):

```
http/
└── api_examples.http
```

Open `http/api_examples.http` and click the green play button next to each request:

**1. Health check:**
```http
GET http://localhost:8080/health
```

**2. JSON prediction:**
```http
POST http://localhost:8080/predict
Content-Type: multipart/form-data; boundary=X

--X
Content-Disposition: form-data; name="image"; filename="image.jpg"
Content-Type: image/jpeg

< ../images/00000.png
--X--
```

Response:
```json
{
  "age": 1.3,
  "gender": 1,
  "gender_name": "male",
  "emotion": "neutral",
  "emotion_confidence": 0.833,
  "emotions": {
    "neutral": 0.833,
    "happiness": 0.039,
    "sadness": 0.068,
    ...
  }
}
```

**3. Image with prediction in headers:**
```http
POST http://localhost:8080/images
Content-Type: multipart/form-data; boundary=X

--X
Content-Disposition: form-data; name="image"; filename="test.jpg"
Content-Type: image/jpeg

< ../images/00001.png
--X--
```

Returns the image with predictions in HTTP headers:
```
X-Prediction-Age: 29.1
X-Prediction-Gender: female
X-Prediction-Gender-Code: 0
X-Prediction-Emotion: happiness
X-Prediction-Emotion-Confidence: 0.998
X-Prediction-Emotions: {"happiness":0.998,"neutral":0.0005,...}
```

### 4. API Usage

**Endpoint**: `POST /predict`

**Request**:
```bash
curl -X POST http://localhost:8080/predict \
  -F "image=@photo.jpg"
```

**Response**:
```json
{
  "age": 28.5,
  "gender": 1,
  "gender_name": "male",
  "emotion": "happiness",
  "emotion_confidence": 0.89,
  "emotions": {
    "happiness": 0.89,
    "neutral": 0.08,
    "surprise": 0.02,
    "anger": 0.005,
    "contempt": 0.003,
    "sadness": 0.002,
    "fear": 0.0005,
    "disgust": 0.0003
  }
}
```

**Health Check**:
```bash
curl http://localhost:8080/health
```

## Model Preparation (Python → Rust)

The model is trained in PyTorch but **runs in Rust**. Here's the workflow:

### Step 1: Train Model (Python)

```bash
cd scripts
python3 train_ffhq_soft.py
```

**Output**: `models/agegenderemo.pt` (state dict, ~36MB)

### Step 2: Export to TorchScript

```bash
cd scripts
python3 export_torchscript.py
```

**What it does**:
- Loads PyTorch model architecture + weights
- Traces model with dummy input
- Saves as TorchScript (self-contained, no Python needed)

**Output**: `models/agegenderemo_traced.pt` (TorchScript, ~36MB)

### Step 3: Use in Rust

Rust server loads TorchScript directly:

```rust
// src/model.rs
let model = tch::CModule::load_on_device(model_path, device)?;
```

**No Python dependency at runtime!** ✅

## Key Features

### Soft Emotion Labels
Model outputs **probability distribution** over all emotions, not just the winner:
```json
"emotions": {
  "happiness": 0.65,    // Primary
  "neutral": 0.25,      // Secondary
  "surprise": 0.08,     // Tertiary
  ...
}
```

Better captures mixed/subtle emotions.

### Gender Encoding
- Training: `1 = male`, `0 = female`
- Model output: sigmoid probability (>0.5 = male)

### Image Preprocessing
- Resize to 224×224
- RGB normalization (ImageNet stats)
- Handled in Rust (no Python)

## Performance

**Hardware**: CPU (no GPU required)
- Inference: 19ms/image
- Throughput: ~50 images/sec
- Memory: ~100MB

**Accuracy** (FFHQ validation, 10k images):
- Age MAE: 2.02 years
- Gender: 98.39%
- Emotion: 94.26%

## Dependencies

### Rust (Cargo.toml)
```toml
actix-web = "4.11"      # Web framework
tch = "0.22"            # PyTorch bindings
image = "0.25"          # Image processing
serde = "1.0"           # JSON serialization
```

### Python (model prep only)
```bash
pip install torch pillow numpy tqdm
```

## Development

### Testing Python Model
```bash
cd scripts
python3 test_soft.py
```

### Rust Rebuild
```bash
cargo clean
cargo build --release
```

### Logs
```bash
RUST_LOG=info cargo run --release
```

## Model Quantization (Optional)

For smaller models:

```bash
cd scripts
python3 export_fp16.py  # 18MB (half size, same accuracy)
```

Update Rust to load FP16 model - same API, smaller file.

## API Examples

### cURL
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: multipart/form-data" \
  -F "image=@face.jpg"
```

### Python
```python
import requests

with open("face.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8080/predict",
        files={"image": f}
    )
    print(response.json())
```

### JavaScript
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('http://localhost:8080/predict', {
  method: 'POST',
  body: formData
})
.then(r => r.json())
.then(data => console.log(data));
```

## Error Handling

**Invalid image**:
```json
{"error": "Unsupported image format"}
```

**Missing field**:
```json
{"error": "Missing 'image' field in request"}
```

**Model failure**:
```json
{"error": "Prediction failed: <details>"}
```

## Deployment

### Docker (recommended)
```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bullseye-slim
COPY --from=builder /app/target/release/reface-server /usr/local/bin/
COPY models /models
ENV LIBTORCH=/usr/local/lib
CMD ["reface-server"]
```

### Systemd Service
```ini
[Unit]
Description=Reface Server

[Service]
ExecStart=/usr/local/bin/reface-server
Environment="LIBTORCH=/usr/local/lib"
Restart=always

[Install]
WantedBy=multi-user.target
```

## Why Rust?

- **Performance**: 5-10× faster than Python Flask/FastAPI
- **Memory**: Lower footprint, no GIL
- **Deployment**: Single binary, no Python runtime
- **Safety**: No segfaults, memory leaks
- **Concurrency**: Native async, handles 1000s connections

PyTorch for **training**, Rust for **production**. Best of both worlds.

## License

Research and educational use only.

## Dataset

- **FFHQ**: Flickr-Faces-HQ (70k high-quality faces)
- **Labels**: Microsoft Azure Face API
- **Split**: 60k train / 10k validation
