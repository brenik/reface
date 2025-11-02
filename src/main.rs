use actix_multipart::Multipart;
use actix_web::{middleware, web, App, HttpResponse, HttpServer, Result};
use futures_util::TryStreamExt;
use log::{debug, error, info};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::path::Path;

mod model;
use model::AgeGenderEmotionPredictor;

#[derive(Serialize)]
struct PredictionResponse {
    age: f32,
    gender: u8,
    gender_name: String,
    emotion: String,
    emotion_confidence: f32,
    emotions: HashMap<String, f32>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Deserialize)]
struct FaceAttributes {
    age: f32,
    gender: String,
    emotion: HashMap<String, f32>,
}

#[derive(Deserialize)]
struct FaceData {
    #[serde(rename = "faceAttributes")]
    face_attributes: FaceAttributes,
}

fn load_ground_truth(image_path: &str) -> Option<(f32, String, HashMap<String, f32>)> {
    let path = Path::new(image_path);
    let filename = path.file_stem()?.to_str()?;
    
    let json_path = format!(
        "/home/fox/rust/reface/images/01_raw/ffhq-dataset/json/{}.json",
        filename
    );
    
    if !Path::new(&json_path).exists() {
        return None;
    }
    
    let json_content = std::fs::read_to_string(&json_path).ok()?;
    let data: Vec<FaceData> = serde_json::from_str(&json_content).ok()?;
    
    if data.is_empty() {
        return None;
    }
    
    let attrs = &data[0].face_attributes;
    Some((attrs.age, attrs.gender.clone(), attrs.emotion.clone()))
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

    info!("Received image data: {} bytes", image_data.len());

    if image_data.len() < 4 {
        error!("Image data too small (< 4 bytes)");
        return Ok(HttpResponse::BadRequest().json(ErrorResponse {
            error: "Invalid image data (too small)".to_string(),
        }));
    }

    let is_valid_image = matches!(
        &image_data[0..2],
        b"\xFF\xD8" | b"\x89\x50" | b"BM" | b"GI"
    );

    if !is_valid_image {
        error!("Invalid image format - magic bytes: {:02X} {:02X}",
            image_data[0], image_data[1]);
        return Ok(HttpResponse::BadRequest().json(ErrorResponse {
            error: "Unsupported image format (only JPEG, PNG, BMP, GIF supported)".to_string(),
        }));
    }

    debug!("Valid image format detected");

    info!("Acquiring predictor lock...");
    let predictor = predictor.lock().unwrap();

    info!("Running prediction...");
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
                emotion: result.emotion_name.clone(),
                emotion_confidence: result.emotion_confidence,
                emotions: result.emotion_probabilities.clone(),
            };

            info!(
                "RESULT: age={:.1}, gender={}, emotion={} ({:.2}%)",
                result.age,
                gender_name,
                result.emotion_name,
                result.emotion_confidence * 100.0
            );
            info!("=== REQUEST COMPLETE ===\n");

            Ok(HttpResponse::Ok().json(response))
        }
        Err(e) => {
            error!("✗ Prediction failed!");
            error!("Error: {:#}", e);
            error!("=== REQUEST FAILED ===\n");

            Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Prediction failed: {}", e),
            }))
        }
    }
}

async fn health() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({"status": "ok"})))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    let model_path = "/home/fox/rust/reface/models/refacemo/agegenderemo_traced.pt";

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
    info!("  POST /predict - Upload image for prediction");
    info!("  GET  /health  - Health check");
    info!("{}", "=".repeat(80));
    info!("");

    HttpServer::new(move || {
        App::new()
            .app_data(predictor_data.clone())
            .app_data(web::PayloadConfig::new(10 * 1024 * 1024))
            .wrap(middleware::Logger::default())
            .route("/predict", web::post().to(predict))
            .route("/predict_image", web::post().to(predict_image))
            .route("/predict_image/{index}", web::post().to(predict_image_with_index))
            .route("/images", web::post().to(predict_image_multipart))
            .route("/health", web::get().to(health))
    })
        .bind(("0.0.0.0", 8080))?
        .run()
        .await
}

async fn predict_image_multipart(
    mut payload: Multipart,
    predictor: web::Data<Arc<Mutex<AgeGenderEmotionPredictor>>>,
) -> Result<HttpResponse> {
    info!("=== NEW IMAGE PREDICTION REQUEST ===");

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

    info!("Received image: {} ({} bytes)", original_filename, image_data.len());

    let predictor = predictor.lock().unwrap();

    match predictor.predict(&image_data) {
        Ok(result) => {
            info!("✓ Prediction successful!");

            let gender_name = if result.gender == 1 { "male" } else { "female" };

            let result_filename = format!(
                "age{:.0}_{}_{}_{:.0}pct_{}",
                result.age,
                gender_name,
                result.emotion_name,
                result.emotion_confidence * 100.0,
                original_filename
            );

            info!(
                "RESULT: age={:.1}, gender={}, emotion={} ({:.2}%)",
                result.age, gender_name, result.emotion_name,
                result.emotion_confidence * 100.0
            );
            info!("=== REQUEST COMPLETE ===\n");

            Ok(HttpResponse::Ok()
                .insert_header(("Content-Type", "image/jpeg"))
                .insert_header(("Content-Disposition",
                                format!("inline; filename=\"{}\"", result_filename)))
                .insert_header(("X-Prediction-Age", result.age.to_string()))
                .insert_header(("X-Prediction-Gender", gender_name))
                .insert_header(("X-Prediction-Gender-Code", result.gender.to_string()))
                .insert_header(("X-Prediction-Emotion", result.emotion_name.clone()))
                .insert_header(("X-Prediction-Emotion-Confidence",
                                result.emotion_confidence.to_string()))
                .body(image_data))
        }
        Err(e) => {
            error!("✗ Prediction failed: {}", e);
            error!("=== REQUEST FAILED ===\n");

            Ok(HttpResponse::InternalServerError()
                .content_type("text/plain")
                .body(format!("Prediction failed: {}", e)))
        }
    }
}

async fn predict_image(
    mut payload: Multipart,
    predictor: web::Data<Arc<Mutex<AgeGenderEmotionPredictor>>>,
) -> Result<HttpResponse> {
    info!("=== PREDICT_IMAGE REQUEST ===");

    let mut image_data = Vec::new();
    let mut path_string = String::new();

    while let Some(mut field) = payload.try_next().await? {
        if let Some(disposition) = field.content_disposition() {
            if disposition.get_name() == Some("image") {
                while let Some(chunk) = field.try_next().await? {
                    image_data.write_all(&chunk)?;
                }
            } else if disposition.get_name() == Some("path") {
                while let Some(chunk) = field.try_next().await? {
                    path_string.push_str(std::str::from_utf8(&chunk).unwrap_or(""));
                }
            }
        }
    }

    if !image_data.is_empty() {
        info!("Processing uploaded image ({} bytes)", image_data.len());
        return process_image_and_return_both(image_data, None, predictor).await;
    }

    if !path_string.is_empty() {
        info!("Processing from path: {}", path_string);
        let path = std::path::Path::new(&path_string);
        
        if path.is_file() {
            info!("Loading file: {}", path_string);
            match std::fs::read(path) {
                Ok(data) => return process_image_and_return_both(data, Some(path_string.clone()), predictor).await,
                Err(e) => {
                    error!("Failed to read file: {}", e);
                    return Ok(HttpResponse::NotFound().json(ErrorResponse {
                        error: format!("File not found: {}", e),
                    }));
                }
            }
        } else if path.is_dir() {
            info!("Loading first image from directory");
            match find_first_image_in_dir(path) {
                Some(img_path) => {
                    info!("Found: {:?}", img_path);
                    match std::fs::read(&img_path) {
                        Ok(data) => return process_image_and_return_both(data, Some(img_path.to_string_lossy().to_string()), predictor).await,
                        Err(e) => {
                            return Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                                error: format!("Failed to read image: {}", e),
                            }));
                        }
                    }
                }
                None => {
                    return Ok(HttpResponse::NotFound().json(ErrorResponse {
                        error: "No images found in directory".to_string(),
                    }));
                }
            }
        }
    }

    Ok(HttpResponse::BadRequest().json(ErrorResponse {
        error: "No image provided (upload file or specify path)".to_string(),
    }))
}

async fn predict_image_with_index(
    index: web::Path<usize>,
    mut payload: Multipart,
    predictor: web::Data<Arc<Mutex<AgeGenderEmotionPredictor>>>,
) -> Result<HttpResponse> {
    let idx = index.into_inner();
    info!("=== PREDICT_IMAGE WITH INDEX: {} ===", idx);

    let mut path_string = String::new();

    while let Some(mut field) = payload.try_next().await? {
        if let Some(disposition) = field.content_disposition() {
            if disposition.get_name() == Some("path") {
                while let Some(chunk) = field.try_next().await? {
                    path_string.push_str(std::str::from_utf8(&chunk).unwrap_or(""));
                }
            }
        }
    }

    if path_string.is_empty() {
        return Ok(HttpResponse::BadRequest().json(ErrorResponse {
            error: "Missing 'path' field in request".to_string(),
        }));
    }

    info!("Loading image {} from: {}", idx, path_string);
    let path = std::path::Path::new(&path_string);

    if !path.is_dir() {
        return Ok(HttpResponse::BadRequest().json(ErrorResponse {
            error: format!("Path is not a directory: {}", path_string),
        }));
    }

    match find_image_by_index(path, idx) {
        Some(img_path) => {
            info!("Found image at index {}: {:?}", idx, img_path);
            match std::fs::read(&img_path) {
                Ok(data) => process_image_and_return_both(data, Some(img_path.to_string_lossy().to_string()), predictor).await,
                Err(e) => {
                    Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                        error: format!("Failed to read image: {}", e),
                    }))
                }
            }
        }
        None => {
            Ok(HttpResponse::NotFound().json(ErrorResponse {
                error: format!("No image found at index {} in directory", idx),
            }))
        }
    }
}

async fn process_image_and_return_both(
    image_data: Vec<u8>,
    image_path: Option<String>,
    predictor: web::Data<Arc<Mutex<AgeGenderEmotionPredictor>>>,
) -> Result<HttpResponse> {
    let predictor = predictor.lock().unwrap();

    match predictor.predict(&image_data) {
        Ok(result) => {
            info!("✓ Prediction successful!");

            let gender_name = if result.gender == 1 { "male" } else { "female" };

            info!(
                "RESULT: age={:.1}, gender={}, emotion={} ({:.2}%)",
                result.age, gender_name, result.emotion_name,
                result.emotion_confidence * 100.0
            );
            info!("=== REQUEST COMPLETE ===\n");

            let emotions_json = serde_json::to_string(&result.emotion_probabilities)
                .unwrap_or_else(|_| "{}".to_string());

            let mut response = HttpResponse::Ok();
            response.insert_header(("Content-Type", "image/jpeg"));

            // Try to load ground truth
            if let Some(ref path) = image_path {
                if let Some((gt_age, gt_gender, gt_emotions)) = load_ground_truth(path) {
                    info!("Ground truth loaded for comparison");
                    
                    let gt_gender_code = if gt_gender.to_lowercase() == "male" { 1 } else { 0 };
                    
                    response.insert_header((
                        "X-Prediction-Age",
                        format!("{}::{}", result.age, gt_age)
                    ));
                    response.insert_header((
                        "X-Prediction-Gender",
                        format!("{}::{}", gender_name, gt_gender.to_lowercase())
                    ));
                    response.insert_header((
                        "X-Prediction-Gender-Code",
                        format!("{}::{}", result.gender, gt_gender_code)
                    ));
                    
                    let mut emotions_with_gt = HashMap::new();
                    for (emotion, pred_value) in &result.emotion_probabilities {
                        let gt_value = gt_emotions.get(emotion).unwrap_or(&0.0);
                        emotions_with_gt.insert(
                            emotion.clone(),
                            format!("{}::{}", pred_value, gt_value)
                        );
                    }
                    
                    let emotions_gt_json = serde_json::to_string(&emotions_with_gt)
                        .unwrap_or_else(|_| "{}".to_string());
                    
                    response.insert_header(("X-Prediction-Emotions", emotions_gt_json));
                    response.insert_header(("X-Prediction-Emotion", result.emotion_name.clone()));
                    response.insert_header((
                        "X-Prediction-Emotion-Confidence",
                        result.emotion_confidence.to_string()
                    ));
                } else {
                    // No ground truth found
                    response.insert_header(("X-Prediction-Age", result.age.to_string()));
                    response.insert_header(("X-Prediction-Gender", gender_name));
                    response.insert_header(("X-Prediction-Gender-Code", result.gender.to_string()));
                    response.insert_header(("X-Prediction-Emotion", result.emotion_name.clone()));
                    response.insert_header(("X-Prediction-Emotion-Confidence", result.emotion_confidence.to_string()));
                    response.insert_header(("X-Prediction-Emotions", emotions_json));
                }
            } else {
                // No path provided, can't load ground truth
                response.insert_header(("X-Prediction-Age", result.age.to_string()));
                response.insert_header(("X-Prediction-Gender", gender_name));
                response.insert_header(("X-Prediction-Gender-Code", result.gender.to_string()));
                response.insert_header(("X-Prediction-Emotion", result.emotion_name.clone()));
                response.insert_header(("X-Prediction-Emotion-Confidence", result.emotion_confidence.to_string()));
                response.insert_header(("X-Prediction-Emotions", emotions_json));
            }

            Ok(response.body(image_data))
        }
        Err(e) => {
            error!("✗ Prediction failed: {}", e);
            error!("=== REQUEST FAILED ===\n");

            Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Prediction failed: {}", e),
            }))
        }
    }
}

fn find_first_image_in_dir(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext_str = ext.to_str()?.to_lowercase();
                    if matches!(ext_str.as_str(), "jpg" | "jpeg" | "png" | "bmp" | "gif") {
                        return Some(path);
                    }
                }
            }
        }
    }
    None
}

fn find_image_by_index(dir: &std::path::Path, index: usize) -> Option<std::path::PathBuf> {
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
            .collect();
        
        images.sort_by_key(|e| e.path());
        
        images.get(index).map(|e| e.path())
    } else {
        None
    }
}
