"""
EcoAesthetics Backend API
FastAPI server with CNN models for urban sustainability analysis
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import io
import logging
from typing import Dict, Any
import asyncio

from models.ensemble_sustainability_analyzer import EnsembleSustainabilityAnalyzer
from services.image_processor import ImageProcessor
from utils.logger import setup_logger
from utils.json_utils import ensure_json_serializable

# Setup logging
logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EcoAesthetics AI Backend",
    description="Urban sustainability analysis using computer vision and deep learning",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:3001",  # Alternative React port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded once at startup)
analyzer = None
image_processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on server startup"""
    global analyzer, image_processor
    
    logger.info("Starting EcoAesthetics backend server...")
    logger.info("Loading AI models...")
    
    try:
        # Initialize models
        analyzer = EnsembleSustainabilityAnalyzer()
        image_processor = ImageProcessor()
        
        # Load models (this might take a few seconds)
        await analyzer.load_models()
        
        logger.info("✅ All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {str(e)}")
        raise e

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "EcoAesthetics AI Backend is running!",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": analyzer is not None,
        "services": {
            "sustainability_analyzer": analyzer is not None,
            "image_processor": image_processor is not None
        }
    }

@app.post("/analyze-sustainability")
async def analyze_sustainability(file: UploadFile = File(...)):
    """
    Analyze street image for sustainability metrics
    
    Args:
        file: Uploaded image file (JPG, PNG, etc.)
        
    Returns:
        JSON response with sustainability analysis
    """
    
    if not analyzer or not image_processor:
        raise HTTPException(
            status_code=503, 
            detail="AI models not loaded. Please try again in a few moments."
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPG, PNG, etc.)"
        )
    
    try:
        logger.info(f"Processing image: {file.filename}")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Validate image size (optional - resize if too large)
        image = image_processor.preprocess_image(image)
        
        # Run AI analysis
        logger.info("Running sustainability analysis...")
        result = await analyzer.analyze_image(image)
        
        # Add metadata
        result.update({
            "filename": file.filename,
            "image_size": image.size,
            "processing_time": result.get("processing_time", 0),
            "model_version": "1.0.0",
            "timestamp": result.get("timestamp")
        })
        
        logger.info(f"Analysis complete. Score: {result.get('score', 'N/A')}")
        
        # Ensure all data is JSON serializable
        serializable_result = ensure_json_serializable(result)
        
        return JSONResponse(content=serializable_result)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze image: {str(e)}"
        )

@app.post("/analyze-batch")
async def analyze_batch(files: list[UploadFile] = File(...)):
    """
    Analyze multiple images in batch
    
    Args:
        files: List of uploaded image files
        
    Returns:
        JSON response with batch analysis results
    """
    
    if not analyzer or not image_processor:
        raise HTTPException(
            status_code=503,
            detail="AI models not loaded. Please try again in a few moments."
        )
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images per batch"
        )
    
    results = []
    
    for file in files:
        try:
            # Process each image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image = image_processor.preprocess_image(image)
            result = await analyzer.analyze_image(image)
            
            result.update({
                "filename": file.filename,
                "image_size": image.size
            })
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "error": str(e),
                "score": None
            })
    
    batch_result = {
        "batch_results": results,
        "total_images": len(files),
        "successful": len([r for r in results if "error" not in r]),
        "failed": len([r for r in results if "error" in r])
    }
    
    # Ensure all data is JSON serializable
    serializable_batch_result = ensure_json_serializable(batch_result)
    
    return JSONResponse(content=serializable_batch_result)

@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    
    if not analyzer:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "models": analyzer.get_model_info(),
        "capabilities": [
            "Object Detection",
            "Semantic Segmentation", 
            "Sustainability Scoring",
            "Feature Extraction"
        ],
        "supported_formats": ["JPG", "JPEG", "PNG", "BMP", "TIFF"],
        "max_image_size": "4096x4096",
        "categories": [
            "Green Coverage",
            "Walkability", 
            "Transit Access",
            "Car Dependency",
            "Building Efficiency",
            "Infrastructure",
            "Natural Elements"
        ]
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
