use actix_multipart::Multipart;
use actix_web::{middleware, web, App, HttpResponse, HttpServer, Result};
use futures_util::TryStreamExt;
use log::{debug, error, info};
use serde::Serialize;
use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, Mutex};

mod model;
use model::AgeGenderEmotionPredictor;

#[derive(Serialize)]
struct PredictionResponse {
    age: f32,
    gender: u8,
    gender_name: String,
    emotions: HashMap<String, f32>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Serialize)]
struct BatchPredictionItem {
    filename: String,
    age: f32,
    gender: u8,
    gender_name: String,
    emotions: HashMap<String, f32>,
}

async fn predict(
    mut payload: Multipart,
    predictor: web::Data<Arc<Mutex<AgeGenderEmotionPredictor>>>,
) -> Result<HttpResponse> {
    info!("=== NEW PREDICTION REQUEST ===");

    let mut image_data = Vec::new();
    let mut has_file = false;

    info!("Reading multipart data...");
    while let Some(mut field) = payload.try_next().await? {
        let content_type = field.content_disposition();
        info!("Field: {:?}", content_type);

        if let Some(disposition) = content_type {
            if disposition.get_name() == Some("image") {
                has_file = true;
            }
        }

        while let Some(chunk) = field.try_next().await? {
            image_data.write_all(&chunk)?;
        }
    }

    if !has_file {
        error!("No 'image' field found in multipart data");
        return Ok(HttpResponse::BadRequest().json(ErrorResponse {
            error: "Missing 'image' field in request".to_string(),
        }));
    }

    if image_data.is_empty() {
        error!("Image field is empty");
        return Ok(HttpResponse::BadRequest().json(ErrorResponse {
            error: "Image data is empty".to_string(),
        }));
    }

    debug!("Received {} bytes", image_data.len());

    if image_data.len() < 4 {
        error!("Image data too small (< 4 bytes)");
        return Ok(HttpResponse::BadRequest().json(ErrorResponse {
            error: "Invalid image data (too small)".to_string(),
        }));
    }

    if !is_valid_image_format(&image_data) {
        error!("Invalid image format - magic bytes: {:02X} {:02X}",
            image_data[0], image_data[1]);
        return Ok(HttpResponse::BadRequest().json(ErrorResponse {
            error: "Unsupported image format (only JPEG, PNG, BMP, GIF supported)".to_string(),
        }));
    }

    let predictor = match predictor.lock() {
        Ok(p) => p,
        Err(e) => {
            error!("Failed to acquire predictor lock: {}", e);
            return Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Server is busy, please try again".to_string(),
            }));
        }
    };

    match predictor.predict(&image_data) {
        Ok(result) => {
            info!("✓ Prediction successful!");

            let gender_name = if result.gender == 1 {
                "male"
            } else {
                "female"
            };

            let response = PredictionResponse {
                age: result.age,
                gender: result.gender,
                gender_name: gender_name.to_string(),
                emotions: result.emotions.clone(),
            };

            info!("Prediction: age={:.1}, gender={}", result.age, gender_name);

            Ok(HttpResponse::Ok().json(response))
        }
        Err(e) => {
            error!("Prediction failed: {:#}", e);

            Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Prediction failed: {}", e),
            }))
        }
    }
}

fn is_valid_image_format(data: &[u8]) -> bool {
    if data.len() < 2 {
        return false;
    }

    let magic_bytes = &data[0..2];

    magic_bytes == b"\xFF\xD8"  // JPEG
        || magic_bytes == b"\x89\x50"  // PNG
        || magic_bytes == b"BM"        // BMP
        || magic_bytes == b"GI"        // GIF
}

fn detect_image_mime_type(data: &[u8]) -> &'static str {
    if data.len() < 2 {
        return "application/octet-stream";
    }

    match &data[0..2] {
        b"\xFF\xD8" => "image/jpeg",
        b"\x89\x50" => "image/png",
        b"BM" => "image/bmp",
        b"GI" => "image/gif",
        _ => "application/octet-stream",
    }
}

async fn predict_image_multipart(
    mut payload: Multipart,
    predictor: web::Data<Arc<Mutex<AgeGenderEmotionPredictor>>>,
) -> Result<HttpResponse> {
    debug!("New prediction request");

    let mut image_data = Vec::new();
    let mut original_filename = String::from("prediction.jpg");

    while let Some(mut field) = payload.try_next().await? {
        if let Some(disposition) = field.content_disposition() {
            if let Some(filename) = disposition.get_filename() {
                original_filename = filename.to_string();
            }
        }

        while let Some(chunk) = field.try_next().await? {
            image_data.write_all(&chunk)?;
        }
    }

    if image_data.is_empty() {
        error!("No image data received");
        return Ok(HttpResponse::BadRequest()
            .content_type("text/plain")
            .body("No image data received"));
    }

    debug!("Processing: {} ({} bytes)", original_filename, image_data.len());

    let predictor = match predictor.lock() {
        Ok(p) => p,
        Err(e) => {
            error!("Failed to acquire predictor lock: {}", e);
            return Ok(HttpResponse::InternalServerError()
                .content_type("text/plain")
                .body("Server is busy, please try again"));
        }
    };

    match predictor.predict(&image_data) {
        Ok(result) => {
            let gender_name = if result.gender == 1 { "male" } else { "female" };

            let result_filename = format!(
                "age{:.0}_{}_ {}",
                result.age,
                gender_name,
                original_filename
            );

            info!(
                "RESULT: age={:.1}, gender={}",
                result.age, gender_name
            );

            let emotions_json = serde_json::to_string(&result.emotions)
                .unwrap_or_else(|_| "{}".to_string());

            let content_type = detect_image_mime_type(&image_data);

            Ok(HttpResponse::Ok()
                .insert_header(("Content-Type", content_type))
                .insert_header(("Content-Disposition",
                                format!("inline; filename=\"{}\"", result_filename)))
                .insert_header(("X-Prediction-Age", result.age.to_string()))
                .insert_header(("X-Prediction-Gender", gender_name))
                .insert_header(("X-Prediction-Gender-Code", result.gender.to_string()))
                .insert_header(("X-Prediction-Emotions", emotions_json))
                .body(image_data))
        }
        Err(e) => {
            error!("Prediction failed: {}", e);

            Ok(HttpResponse::InternalServerError()
                .content_type("text/plain")
                .body(format!("Prediction failed: {}", e)))
        }
    }
}

async fn health() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({"status": "ok"})))
}

async fn predict_input(
    predictor: web::Data<Arc<Mutex<AgeGenderEmotionPredictor>>>,
) -> Result<HttpResponse> {
    debug!("Batch prediction from input directory");

    let input_dir = "input/";
    let path = std::path::Path::new(input_dir);

    if !path.exists() || !path.is_dir() {
        return Ok(HttpResponse::BadRequest().json(ErrorResponse {
            error: format!("Directory does not exist: {}", input_dir),
        }));
    }

    let images = get_all_images_from_dir(path);

    if images.is_empty() {
        return Ok(HttpResponse::NotFound().json(ErrorResponse {
            error: "No images found in input directory".to_string(),
        }));
    }

    info!("Processing {} images from input/", images.len());

    let mut results = Vec::new();
    let predictor = match predictor.lock() {
        Ok(p) => p,
        Err(e) => {
            error!("Failed to acquire predictor lock: {}", e);
            return Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Server is busy, please try again".to_string(),
            }));
        }
    };

    for (idx, img_path) in images.iter().enumerate() {
        let filename = img_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        debug!("[{}/{}] {}", idx + 1, images.len(), filename);

        match std::fs::read(&img_path) {
            Ok(image_data) => {
                match predictor.predict(&image_data) {
                    Ok(result) => {
                        let gender_name = if result.gender == 1 { "male" } else { "female" };

                        info!("  ✓ age={:.1}, gender={}",
                              result.age, gender_name);

                        results.push(BatchPredictionItem {
                            filename,
                            age: result.age,
                            gender: result.gender,
                            gender_name: gender_name.to_string(),
                            emotions: result.emotions,
                        });
                    }
                    Err(e) => {
                        error!("Failed {}: {}", filename, e);
                    }
                }
            }
            Err(e) => {
                error!("Cannot read {}: {}", filename, e);
            }
        }
    }

    info!("Batch complete: {}/{} successful", results.len(), images.len());

    Ok(HttpResponse::Ok().json(results))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    let model_path = "models/agegenderemo_traced.pt";

    info!("{}", "=".repeat(80));
    info!("AGE/GENDER/EMOTION PREDICTION SERVER");
    info!("{}", "=".repeat(80));
    info!("Loading model from: {}", model_path);

    let predictor = match AgeGenderEmotionPredictor::new(model_path) {
        Ok(p) => {
            info!("✓ Model loaded successfully");
            Arc::new(Mutex::new(p))
        }
        Err(e) => {
            error!("✗ Failed to load model: {:#}", e);
            std::process::exit(1);
        }
    };

    let predictor_data = web::Data::new(predictor);

    info!("");
    info!("Starting server on http://0.0.0.0:8080");
    info!("Endpoints:");
    info!("  POST /predict - Upload image for JSON prediction");
    info!("  POST /images  - Upload image, get image back with prediction in headers");
    info!("  GET  /health  - Health check");
    info!("{}", "=".repeat(80));
    info!("");

    HttpServer::new(move || {
        App::new()
            .app_data(predictor_data.clone())
            .app_data(web::PayloadConfig::new(10 * 1024 * 1024))
            .wrap(middleware::Logger::default())
            .route("/predict", web::post().to(predict))
            .route("/images", web::post().to(predict_image_multipart))
            .route("/predict_input", web::get().to(predict_input))
            .route("/health", web::get().to(health))
    })
        .bind(("0.0.0.0", 8080))?
        .run()
        .await
}

fn get_all_images_from_dir(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut images: Vec<_> = entries
            .flatten()
            .filter(|e| {
                let path = e.path();
                path.is_file() && path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|s| matches!(s.to_lowercase().as_str(), "jpg" | "jpeg" | "png" | "bmp" | "gif"))
                    .unwrap_or(false)
            })
            .map(|e| e.path())
            .collect();

        images.sort();
        images
    } else {
        Vec::new()
    }
}
