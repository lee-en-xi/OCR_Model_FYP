from flask import Flask, request, jsonify
from paddleocr import PPStructureV3
from PIL import Image
import numpy as np
import io
import logging
import os
from datetime import datetime
import cv2
import json

# Configuration
class Config:
    DEBUG = False  # Production mode
    PORT = int(os.environ.get("PORT", 10000))  # Render assigns PORT
    HOST = "0.0.0.0"
    MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB
    OCR_CONFIG = {
        "device": os.environ.get("OCR_DEVICE", "cpu"),
        "text_detection_model_name": os.environ.get("TEXT_DETECTION_MODEL", "PP-OCRv5_mobile_det"),
        "text_recognition_model_dir": os.environ.get(
            "TEXT_RECOGNITION_MODEL_DIR",
            os.path.join(os.path.dirname(__file__), "finetuned_PP-OCRv5_mobile_rec_model")
        ),
        "text_recognition_model_name": os.environ.get("TEXT_RECOGNITION_MODEL", "PP-OCRv5_mobile_rec"),
    }

app = Flask(__name__)
app.config.from_object(Config)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    model_dir = Config.OCR_CONFIG["text_recognition_model_dir"]
    if not os.path.exists(model_dir):
        logger.error(f"Model directory {model_dir} does not exist")
        raise FileNotFoundError(f"Model directory {model_dir} not found")
    logger.info(f"Model directory found: {model_dir}")
    ocr = PPStructureV3(**app.config["OCR_CONFIG"])
    logger.info("✅ PaddleOCR PPStructureV3 initialized successfully")
except Exception as e:
    logger.error(f"❌ PaddleOCR PPStructureV3 initialization failed: {e}", exc_info=True)
    ocr = None

def convert_paddleocr_result(raw_result):
    output = []
    if not raw_result:
        return output
    
    for result_obj in raw_result:
        try:
            if isinstance(result_obj, dict):
                parsing_res = result_obj.get('parsing_res_list', [])
                overall_res = result_obj.get('overall_ocr_res', {})
            else:
                parsing_res = getattr(result_obj, 'parsing_res_list', [])
                overall_res = getattr(result_obj, 'overall_ocr_res', {})
            
            for score, box, text in zip(overall_res.get('rec_scores', []), overall_res.get('rec_boxes', []), overall_res.get('rec_texts', [])):
                try:
                    output.append({
                        "text": text,
                        "confidence": float(score),
                        "coordinates": box.tolist() if isinstance(box, np.ndarray) else box,
                        "label": "text"
                    })
                except Exception as block_error:
                    logger.error(f"Error processing block: {block_error}", exc_info=True)
                    continue
        except Exception as e:
            logger.error(f"Error processing result: {e}", exc_info=True)
            continue
    
    logger.info(f"Processed {len(output)} text blocks")
    return output

def process_image_file(file):
    try:
        img_bytes = file.read()
        np_array = np.frombuffer(img_bytes, np.uint8)
        img_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if img_array is None:
            logger.warning("OpenCV failed to decode image. Falling back to PIL.")
            img = Image.open(io.BytesIO(img_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np }}">
        logger.info(f"Image processed. Shape: {img_array.shape}, Type: {img_array.dtype}")
        return img_array
    except Exception as e:
        logger.error(f"Image processing failed: {e}", exc_info=True)
        raise

@app.route('/api/ocr', methods=['POST'])
def ocr_route():
    if not ocr:
        return jsonify({
            "success": False,
            "error": "OCR engine not initialized",
            "result": None,
            "timestamp": datetime.now().isoformat()
        }), 500
    
    try:
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded",
                "result": None,
                "timestamp": datetime.now().isoformat()
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected",
                "result": None,
                "timestamp": datetime.now().isoformat()
            }), 400

        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        logger.info(f"File size: {file_size} bytes")
        if file_size > Config.MAX_FILE_SIZE:
            logger.error(f"File size {file_size} exceeds limit {Config.MAX_FILE_SIZE}")
            return jsonify({
                "success": False,
                "error": f"File size exceeds limit of {Config.MAX_FILE_SIZE / (1024 * 1024)}MB",
                "result": None,
                "timestamp": datetime.now().isoformat()
            }), 400
        file.seek(0)

        logger.info(f"Processing file: {file.filename}")
        img_array = process_image_file(file)
        result = ocr.predict(img_array)
        logger.info(f"Raw OCR result type: {type(result)}")
        
        processed_result = convert_paddleocr_result(result)
        
        if not processed_result:
            logger.warning(f"No content detected in {file.filename}")
            return jsonify({
                "success": True,
                "message": "No text/content detected in image",
                "result": [],
                "timestamp": datetime.now().isoformat()
            })

        return jsonify({
            "success": True,
            "message": "OCR processing completed",
            "result": processed_result,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in OCR route: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": str(e),
            "result": None,
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/')
def health_check():
    return jsonify({
        "status": "ready" if ocr else "unavailable",
        "ocr_initialized": bool(ocr),
        "service": "paddle-ocr-api",
        "version": "1.0",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    if app.config["DEBUG"]:
        app.run(host=app.config["HOST"], port=app.config["PORT"], debug=True)
    else:
        logger.info("Production mode: Use Gunicorn to run the app")
