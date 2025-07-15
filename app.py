import os
import time
import logging
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from paddleocr import PPStructureV3
import numpy as np
import cv2
import json

# Configuration
class Config:
    DEBUG = os.environ.get("FLASK_ENV", "production") != "production"
    PORT = int(os.environ.get("PORT", 10000))  # Render uses port 10000
    HOST = "0.0.0.0"
    OCR_TIMEOUT = 300  # 5 minutes timeout for OCR processing
    OCR_CONFIG = {
        "device": "cpu",
        "text_detection_model_name": "PP-OCRv5_mobile_det",
        "text_recognition_model_dir": os.path.join(
            os.path.dirname(__file__), 
            "finetuned_PP-OCRv5_mobile_rec_model"
        ),
        "text_recognition_model_name": "PP-OCRv5_mobile_rec",
    }

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Global OCR engine instance
ocr = None

def initialize_ocr():
    """Initialize the OCR engine with error handling"""
    global ocr
    try:
        ocr = PPStructureV3(**app.config["OCR_CONFIG"])
        logger.info("✅ PaddleOCR initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ OCR initialization failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

@app.before_first_request
def before_first_request():
    """Initialize OCR before first request"""
    if not initialize_ocr():
        logger.error("Failed to initialize OCR engine")

def validate_image_file(file):
    """Validate the uploaded image file"""
    if not file or file.filename == '':
        return False, "No file uploaded or empty filename"
    
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    if '.' not in file.filename or \
       file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return False, "Invalid file type"
    
    return True, ""

def process_image(file_stream):
    """Process image file into numpy array"""
    try:
        img_bytes = file_stream.read()
        np_array = np.frombuffer(img_bytes, np.uint8)
        img_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if img_array is None:
            raise ValueError("OpenCV failed to decode image")
            
        logger.info(f"Image processed. Shape: {img_array.shape}")
        return img_array
        
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise

def format_ocr_result(raw_result):
    """Convert PaddleOCR result to JSON-serializable format"""
    results = []
    if not raw_result:
        return results
    
    for item in raw_result:
        try:
            # Handle both dict and object results
            if isinstance(item, dict):
                data = item
            else:
                data = {k: getattr(item, k) for k in dir(item) if not k.startswith('_')}
            
            if 'overall_ocr_res' in data:
                res = data['overall_ocr_res']
                for text, score, box in zip(
                    res.get('rec_texts', []),
                    res.get('rec_scores', []),
                    res.get('rec_boxes', [])
                ):
                    results.append({
                        "text": text,
                        "confidence": float(score),
                        "coordinates": box.tolist() if hasattr(box, 'tolist') else box,
                        "label": "text"
                    })
                    
        except Exception as e:
            logger.error(f"Error processing OCR item: {str(e)}")
            continue
    
    return results

@app.route('/api/ocr', methods=['POST'])
def ocr_endpoint():
    """Main OCR processing endpoint"""
    start_time = time.time()
    
    if not ocr:
        return jsonify({
            "success": False,
            "error": "OCR engine not initialized",
            "timestamp": datetime.utcnow().isoformat()
        }), 500
    
    # Validate request
    if 'file' not in request.files:
        return jsonify({
            "success": False,
            "error": "No file uploaded",
            "timestamp": datetime.utcnow().isoformat()
        }), 400
        
    file = request.files['file']
    is_valid, error_msg = validate_image_file(file)
    if not is_valid:
        return jsonify({
            "success": False,
            "error": error_msg,
            "timestamp": datetime.utcnow().isoformat()
        }), 400
    
    try:
        # Process image
        img_array = process_image(file)
        
        # Check timeout
        if time.time() - start_time > app.config["OCR_TIMEOUT"] / 2:
            raise TimeoutError("Image processing took too long")
        
        # Perform OCR
        raw_result = ocr.predict(img_array)
        
        # Check timeout again
        if time.time() - start_time > app.config["OCR_TIMEOUT"]:
            raise TimeoutError("OCR processing timeout")
        
        # Format results
        processed_results = format_ocr_result(raw_result)
        
        if not processed_results:
            return jsonify({
                "success": True,
                "message": "No text detected",
                "result": [],
                "timestamp": datetime.utcnow().isoformat()
            })
            
        return jsonify({
            "success": True,
            "result": processed_results,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except TimeoutError as e:
        logger.error(f"Processing timeout: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Processing timeout",
            "timestamp": datetime.utcnow().isoformat()
        }), 408
        
    except Exception as e:
        logger.error(f"OCR processing error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"Processing error: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    status = {
        "status": "ready" if ocr else "unavailable",
        "ocr_initialized": bool(ocr),
        "service": "paddle-ocr-api",
        "version": "1.0",
        "timestamp": datetime.utcnow().isoformat()
    }
    return jsonify(status)

# For local development
if __name__ == '__main__':
    initialize_ocr()
    app.run(
        host=app.config["HOST"],
        port=app.config["PORT"],
        debug=app.config["DEBUG"]
    )
