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
    emotion: String,
    emotion_confidence: f32,
    emotions: HashMap<String, f32>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
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

            let emotions_json = serde_json::to_string(&result.emotion_probabilities)
                .unwrap_or_else(|_| "{}".to_string());

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
                .insert_header(("X-Prediction-Emotions", emotions_json))
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

async fn health() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({"status": "ok"})))
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
            .route("/health", web::get().to(health))
    })
        .bind(("0.0.0.0", 8080))?
        .run()
        .await
}
