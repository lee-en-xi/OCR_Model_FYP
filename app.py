from flask import Flask, request, jsonify
from paddleocr import PPStructureV3
from PIL import Image
import numpy as np
import io
import logging
import os
import traceback
from datetime import datetime
import cv2
import json  # Added missing import

# Configuration
class Config:
    DEBUG = True
    PORT = int(os.environ.get("PORT", 5000))
    HOST = "0.0.0.0"
    OCR_CONFIG = {
        "device": "cpu",
        "text_detection_model_name": "PP-OCRv5_mobile_det",
       "text_recognition_model_dir": "finetuned_PP-OCRv5_mobile_rec_model",
        "text_recognition_model_name": "PP-OCRv5_mobile_rec",
    }
    

app = Flask(__name__)
app.config.from_object(Config)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    ocr = PPStructureV3(**app.config["OCR_CONFIG"])
    logger.info("✅ PaddleOCR PPStructureV3 initialized successfully")
except Exception as e:
    logger.error(f"❌ PaddleOCR PPStructureV3 initialization failed: {e}", exc_info=True)
    ocr = None

def convert_paddleocr_result(raw_result):
    """
    Updated converter that handles both dictionary and object results
    """
    output = []
    if not raw_result:
        return output
    
    for result_obj in raw_result:
        try:
            # Handle both dictionary and object results
            if isinstance(result_obj, dict):
                # Dictionary format
                parsing_res = result_obj.get('parsing_res_list', [])
                overall_res = result_obj.get('overall_ocr_res', {})
            else:
                # Object format - use getattr
                parsing_res = getattr(result_obj, 'parsing_res_list', [])
                overall_res = getattr(result_obj, 'overall_ocr_res', {})
            
            for score, box, text in zip(overall_res.get('rec_scores', []), overall_res.get('rec_boxes', []), overall_res.get('rec_texts', [])):
                try: 
                    output.append({
                    "text": text,
                    "confidence": float(score),
                    # fix json serialization issue with numpy arrays
                    "coordinates": box.tolist() if isinstance(box, np.ndarray) else box,
                    "label": "text"
                })
                except Exception as block_error:
                    logger.error(f"Error processing block: {block_error}", exc_info=True)
                    continue
            
                    
        except Exception as e:
            logger.error(f"Error processing result: {e}", exc_info=True)
            continue
    
    return output

def process_image_file(file):
    """Robust image processing with multiple fallbacks"""
    try:
        img_bytes = file.read()
        
        # Try OpenCV first
        np_array = np.frombuffer(img_bytes, np.uint8)
        img_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if img_array is None:
            logger.warning("OpenCV failed to decode image. Falling back to PIL.")
            img = Image.open(io.BytesIO(img_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
            
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

        logger.info(f"Processing file: {file.filename}")
        
        img_array = process_image_file(file)
        
        # Perform OCR
        result = ocr.predict(img_array)
        logger.info(f"Raw OCR result type: {type(result)}")
        
        # Convert to JSON-serializable format for logging
        try:
            result_json = json.dumps(result, indent=2, default=str)
            logger.info(f"Full OCR result structure: {result_json}")
        except Exception as e:
            logger.error(f"Could not serialize OCR result: {e}")
        
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
    app.run(host=app.config["HOST"], port=app.config["PORT"], debug=app.config["DEBUG"])