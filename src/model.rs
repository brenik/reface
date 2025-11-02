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
    pub emotion_name: String,
    pub emotion_confidence: f32,
    pub emotion_probabilities: HashMap<String, f32>,
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
        info!("  [1/4] Preprocessing image ({} bytes)...", image_data.len());
        let tensor = self.preprocess_image(image_data)?;
        info!("  ✓ Preprocessed tensor shape: {:?}", tensor.size());

        info!("  [2/4] Running model inference...");

        let outputs = self
            .model
            .forward_is(&[tch::IValue::Tensor(tensor)])
            .context("Model inference failed")?;

        info!("  ✓ Model returned outputs");
        info!("  RAW MODEL OUTPUT: {:?}", outputs);
        debug!("  Output type: {:?}", outputs);

        info!("  [3/4] Parsing predictions...");
        let age = self.parse_age(&outputs)?;
        info!("  ✓ Age: {:.1}", age);

        let gender = self.parse_gender(&outputs)?;
        info!("  ✓ Gender: {} ({})", gender, if gender == 1 { "female" } else { "male" });

        info!("  [4/4] Parsing emotion with probabilities...");
        let (emotion_name, emotion_confidence, emotion_probabilities) = self.parse_emotion(&outputs)?;
        info!("  ✓ Emotion: {} ({:.2}%)", emotion_name, emotion_confidence * 100.0);

        Ok(PredictionResult {
            age,
            gender,
            emotion_name,
            emotion_confidence,
            emotion_probabilities,
        })
    }

    fn preprocess_image(&self, image_data: &[u8]) -> Result<Tensor> {
        debug!("    Decoding image...");
        let img = image::load_from_memory(image_data)
            .context("Failed to decode image")?;

        debug!("    Original size: {}x{}", img.width(), img.height());

        debug!("    Resizing to 224x224...");
        let img = img.resize_exact(224, 224, FilterType::Lanczos3);
        let img = img.to_rgb8();

        debug!("    Converting to tensor and normalizing...");
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

        debug!("    Creating tensor [1, 3, 224, 224]...");
        let tensor = Tensor::from_slice(&tensor_data)
            .reshape(&[1, 3, 224, 224])
            .to_device(self.device);

        debug!("    Tensor device: {:?}", tensor.device());
        debug!("    Tensor shape: {:?}", tensor.size());
        debug!("    Tensor dtype: {:?}", tensor.kind());

        Ok(tensor)
    }

    fn parse_age(&self, outputs: &tch::IValue) -> Result<f32> {
        debug!("    Extracting age from tuple output...");

        let tuple = match outputs {
            tch::IValue::Tuple(t) => t,
            _ => anyhow::bail!("Output is not a tuple, got: {:?}", outputs),
        };

        debug!("    Tuple length: {}", tuple.len());

        if tuple.len() != 3 {
            anyhow::bail!("Expected 3 outputs, got {}", tuple.len());
        }

        let age_tensor = match &tuple[0] {
            tch::IValue::Tensor(t) => t,
            _ => anyhow::bail!("Age output is not a tensor"),
        };

        debug!("    Age tensor shape: {:?}", age_tensor.size());

        let age_value = f32::try_from(age_tensor)
            .context("Failed to extract age")?;

        debug!("    Raw age value: {:.2}", age_value);

        let clamped_age = age_value.max(0.0).min(100.0);
        debug!("    Clamped age: {:.2}", clamped_age);

        Ok(clamped_age)
    }

    fn parse_gender(&self, outputs: &tch::IValue) -> Result<u8> {
        debug!("    Extracting gender from tuple output...");

        let tuple = match outputs {
            tch::IValue::Tuple(t) => t,
            _ => anyhow::bail!("Output is not a tuple"),
        };

        let gender_tensor = match &tuple[1] {
            tch::IValue::Tensor(t) => t,
            _ => anyhow::bail!("Gender output is not a tensor"),
        };

        debug!("    Gender tensor shape: {:?}", gender_tensor.size());

        let gender_prob = f32::try_from(gender_tensor)
            .context("Failed to extract gender")?;

        debug!("    Gender probability: {:.4}", gender_prob);

        let gender = if gender_prob > 0.5 { 1 } else { 0 };
        debug!("    Gender class: {} ({})", gender, if gender == 1 { "male" } else { "female" });

        Ok(gender)
    }

    fn parse_emotion(&self, outputs: &tch::IValue) -> Result<(String, f32, HashMap<String, f32>)> {
        debug!("    Extracting emotion from tuple output...");

        let tuple = match outputs {
            tch::IValue::Tuple(t) => t,
            _ => anyhow::bail!("Output is not a tuple"),
        };

        let emotion_tensor = match &tuple[2] {
            tch::IValue::Tensor(t) => t,
            _ => anyhow::bail!("Emotion output is not a tensor"),
        };

        debug!("    Emotion tensor shape: {:?}", emotion_tensor.size());

        let emotion_tensor = emotion_tensor.squeeze_dim(0);
        debug!("    Squeezed tensor shape: {:?}", emotion_tensor.size());

        let probs_vec: Vec<f32> = emotion_tensor.try_into()
            .context("Failed to convert emotion tensor to vector")?;

        debug!("    Emotion probabilities (soft labels):");
        let mut emotion_map = HashMap::new();
        for (i, prob) in probs_vec.iter().enumerate() {
            let emotion_name = EMOTION_NAMES[i];
            emotion_map.insert(emotion_name.to_string(), *prob);
            debug!("      {}: {:.2}%", emotion_name, prob * 100.0);
        }

        let (max_idx, max_prob) = probs_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .context("No emotions detected")?;

        let emotion_name = EMOTION_NAMES
            .get(max_idx)
            .ok_or_else(|| anyhow::anyhow!("Invalid emotion index"))?
            .to_string();

        debug!("    Best emotion: {} (index {}, conf {:.2}%)",
            emotion_name, max_idx, max_prob * 100.0);

        Ok((emotion_name, *max_prob, emotion_map))
    }
}
