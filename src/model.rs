use anyhow::{Context, Result};
use image::imageops::FilterType;
use log::{debug, info};
use std::collections::HashMap;
use tch::{Device, Tensor};

const EMOTION_NAMES: [&str; 8] = [
    "anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise",
];

pub struct PredictionResult {
    pub age: f32,
    pub gender: u8,
    pub emotions: HashMap<String, f32>,
}

pub struct AgeGenderEmotionPredictor {
    model: tch::CModule,
    device: Device,
}

impl AgeGenderEmotionPredictor {
    pub fn new(model_path: &str) -> Result<Self> {
        let device = if tch::Cuda::is_available() {
            info!("CUDA available, using GPU");
            Device::Cuda(0)
        } else {
            info!("CUDA not available, using CPU");
            Device::Cpu
        };

        info!("Loading PyTorch model...");
        let model = tch::CModule::load_on_device(model_path, device)
            .context("Failed to load model")?;

        info!("Model loaded on device: {:?}", device);

        Ok(Self { model, device })
    }

    pub fn predict(&self, image_data: &[u8]) -> Result<PredictionResult> {
        debug!("Preprocessing image ({} bytes)", image_data.len());
        let tensor = self.preprocess_image(image_data)?;

        debug!("Running model inference");
        let outputs = self
            .model
            .forward_is(&[tch::IValue::Tensor(tensor)])
            .context("Model inference failed")?;

        let age = self.parse_age(&outputs)?;
        let gender = self.parse_gender(&outputs)?;
        let emotions = self.parse_emotions(&outputs)?;

        debug!("Prediction complete: age={:.1}, gender={}", age, gender);

        Ok(PredictionResult {
            age,
            gender,
            emotions,
        })
    }

    fn preprocess_image(&self, image_data: &[u8]) -> Result<Tensor> {
        let img = image::load_from_memory(image_data)
            .context("Failed to decode image")?;

        let img = img.resize_exact(224, 224, FilterType::Lanczos3);
        let img = img.to_rgb8();

        let mut tensor_data = Vec::with_capacity(3 * 224 * 224);

        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];

        for c in 0..3 {
            for y in 0..224 {
                for x in 0..224 {
                    let pixel = img.get_pixel(x, y);
                    let value = pixel[c] as f32 / 255.0;
                    let normalized = (value - mean[c]) / std[c];
                    tensor_data.push(normalized);
                }
            }
        }

        let tensor = Tensor::from_slice(&tensor_data)
            .reshape(&[1, 3, 224, 224])
            .to_device(self.device);

        Ok(tensor)
    }

    fn parse_age(&self, outputs: &tch::IValue) -> Result<f32> {
        let tuple = match outputs {
            tch::IValue::Tuple(t) => t,
            _ => anyhow::bail!("Output is not a tuple"),
        };

        if tuple.len() != 10 {
            anyhow::bail!("Expected 10 outputs, got {}", tuple.len());
        }

        let age_tensor = match &tuple[0] {
            tch::IValue::Tensor(t) => t,
            _ => anyhow::bail!("Age output is not a tensor"),
        };

        let age_value = f32::try_from(age_tensor)
            .context("Failed to extract age")?;

        let clamped_age = age_value.max(0.0).min(100.0);

        Ok(clamped_age)
    }

    fn parse_gender(&self, outputs: &tch::IValue) -> Result<u8> {
        let tuple = match outputs {
            tch::IValue::Tuple(t) => t,
            _ => anyhow::bail!("Output is not a tuple"),
        };

        let gender_tensor = match &tuple[1] {
            tch::IValue::Tensor(t) => t,
            _ => anyhow::bail!("Gender output is not a tensor"),
        };

        let gender_prob = f32::try_from(gender_tensor)
            .context("Failed to extract gender")?;

        let gender = if gender_prob > 0.5 { 1 } else { 0 };

        Ok(gender)
    }

    fn parse_emotions(&self, outputs: &tch::IValue) -> Result<HashMap<String, f32>> {
        let tuple = match outputs {
            tch::IValue::Tuple(t) => t,
            _ => anyhow::bail!("Output is not a tuple"),
        };

        if tuple.len() != 10 {
            anyhow::bail!("Expected 10 outputs, got {}", tuple.len());
        }

        let mut emotions = HashMap::new();

        for i in 0..8 {
            let tensor = match &tuple[i + 2] {
                tch::IValue::Tensor(t) => t,
                _ => anyhow::bail!("Emotion output {} is not a tensor", i),
            };

            let prob = f32::try_from(tensor)
                .with_context(|| format!("Failed to extract emotion {}", i))?;

            let name = EMOTION_NAMES[i].to_string();
            emotions.insert(name, prob);
        }

        Ok(emotions)
    }
}
