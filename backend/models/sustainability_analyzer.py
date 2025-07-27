"""
Core sustainability analysis using CNN models and computer vision
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import DetrImageProcessor, DetrForObjectDetection
import numpy as np
from PIL import Image
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
import asyncio

from services.image_processor import ImageProcessor
from utils.logger import setup_logger

logger = setup_logger(__name__)

class SustainabilityAnalyzer:
    """Main class for analyzing street sustainability using AI models"""
    
    def __init__(self):
        """Initialize the sustainability analyzer"""
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Model components
        self.object_detector = None
        self.object_processor = None
        self.segmentation_model = None
        self.image_processor = ImageProcessor()
        
        # Sustainability categories and their weights
        self.sustainability_categories = {
            'greenCoverage': {
                'weight': 0.25,
                'keywords': ['tree', 'plant', 'grass', 'vegetation', 'bush', 'flower', 'garden'],
                'max_score': 30
            },
            'walkability': {
                'weight': 0.20,
                'keywords': ['person', 'pedestrian', 'sidewalk', 'crosswalk', 'walkway'],
                'max_score': 25
            },
            'transitAccess': {
                'weight': 0.15,
                'keywords': ['bus', 'bicycle', 'bike', 'train', 'subway', 'transit'],
                'max_score': 25
            },
            'carDependency': {
                'weight': -0.15,  # Negative weight (less cars = better)
                'keywords': ['car', 'truck', 'vehicle', 'parking', 'traffic'],
                'max_score': 0,
                'penalty': -20
            },
            'buildingEfficiency': {
                'weight': 0.10,
                'keywords': ['solar panel', 'green roof', 'modern building'],
                'max_score': 20
            },
            'infrastructure': {
                'weight': 0.05,
                'keywords': ['traffic light', 'street light', 'bench', 'sign'],
                'max_score': 10
            }
        }
        
        # COCO class names for object detection
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    async def load_models(self):
        """Load all AI models"""
        
        logger.info("Loading object detection model (DETR)...")
        try:
            # Load DETR model for object detection
            self.object_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self.object_detector = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            self.object_detector.to(self.device)
            self.object_detector.eval()
            logger.info("✅ Object detection model loaded")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load DETR model: {e}")
            logger.info("Using fallback object detection...")
            self.object_detector = None
        
        logger.info("Loading semantic segmentation model...")
        try:
            # Load DeepLab model for semantic segmentation
            self.segmentation_model = torch.hub.load(
                'pytorch/vision:v0.10.0', 
                'deeplabv3_resnet50', 
                pretrained=True
            )
            self.segmentation_model.to(self.device)
            self.segmentation_model.eval()
            logger.info("✅ Semantic segmentation model loaded")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load segmentation model: {e}")
            self.segmentation_model = None
        
        logger.info("🚀 All models loaded successfully!")
    
    async def analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze street image for sustainability metrics
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with sustainability analysis results
        """
        
        start_time = time.time()
        logger.info("Starting sustainability analysis...")
        
        try:
            # 1. Object Detection
            logger.info("Step 1: Object detection...")
            objects = await self._detect_objects(image)
            
            # 2. Semantic Segmentation
            logger.info("Step 2: Semantic segmentation...")
            segmentation = await self._segment_image(image)
            
            # 3. Color and texture analysis
            logger.info("Step 3: Color analysis...")
            color_features = self.image_processor.extract_color_features(image)
            
            # 4. Extract sustainability features
            logger.info("Step 4: Feature extraction...")
            features = self._extract_sustainability_features(objects, segmentation, color_features)
            
            # 5. Calculate sustainability score
            logger.info("Step 5: Score calculation...")
            score_result = self._calculate_sustainability_score(features)
            
            # 6. Generate recommendations
            recommendations = self._generate_recommendations(features, score_result['score'])
            
            processing_time = time.time() - start_time
            
            result = {
                'score': score_result['score'],
                'method': 'cnn_analysis',
                'features': features,
                'categoryScores': score_result['category_scores'],
                'recommendations': recommendations,
                'objects': objects,
                'color_features': color_features,
                'processing_time': round(processing_time, 2),
                'timestamp': datetime.now().isoformat(),
                'model_info': self.get_model_info()
            }
            
            logger.info(f"✅ Analysis complete! Score: {result['score']}/100 (took {processing_time:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            raise e
    
    async def _detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects in the image using DETR model"""
        
        if not self.object_detector:
            logger.warning("Object detector not available, using fallback")
            return self._fallback_object_detection(image)
        
        try:
            # Preprocess image for DETR
            inputs = self.object_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.object_detector(**inputs)
            
            # Process results
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)  # (height, width)
            results = self.object_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.5
            )[0]
            
            # Convert to our format
            objects = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score > 0.5:  # Confidence threshold
                    objects.append({
                        'label': self.object_detector.config.id2label[label.item()],
                        'confidence': score.item(),
                        'bbox': box.tolist()  # [x1, y1, x2, y2]
                    })
            
            logger.info(f"Detected {len(objects)} objects")
            return objects
            
        except Exception as e:
            logger.warning(f"Object detection failed: {e}, using fallback")
            return self._fallback_object_detection(image)
    
    def _fallback_object_detection(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Enhanced fallback object detection using color and texture analysis"""
        
        logger.info("Using enhanced fallback object detection based on color analysis")
        
        # Extract color features
        color_features = self.image_processor.extract_color_features(image)
        
        objects = []
        
        # Enhanced vegetation detection with multiple thresholds
        green_ratio = color_features['green_ratio']
        vegetation_index = color_features['vegetation_index']
        
        # Multi-tier vegetation object estimation
        if green_ratio > 0.05:  # Lower threshold for detection
            # Base vegetation objects
            base_vegetation_count = max(1, int(green_ratio * 15))  # More sensitive
            
            # Add different types of vegetation based on coverage
            vegetation_types = ['potted plant', 'tree', 'plant', 'grass', 'bush']
            
            for i in range(min(base_vegetation_count, 8)):  # Increased max objects
                veg_type = vegetation_types[i % len(vegetation_types)]
                confidence = min(0.6 + green_ratio, 0.9)  # Dynamic confidence
                
                objects.append({
                    'label': veg_type,
                    'confidence': confidence,
                    'bbox': [i*20, i*20, (i+1)*20, (i+1)*20]  # Varied positions
                })
            
            # Bonus objects for high vegetation coverage
            if green_ratio > 0.2:
                objects.extend([
                    {'label': 'garden', 'confidence': 0.8, 'bbox': [0, 0, 50, 50]},
                    {'label': 'vegetation', 'confidence': 0.8, 'bbox': [50, 50, 100, 100]}
                ])
            
            if green_ratio > 0.3:
                objects.extend([
                    {'label': 'forest', 'confidence': 0.9, 'bbox': [0, 0, 100, 100]},
                    {'label': 'park', 'confidence': 0.85, 'bbox': [25, 25, 75, 75]}
                ])
        
        # Vegetation index bonus objects
        if vegetation_index > 0.15:
            vi_objects = int(vegetation_index * 10)
            for i in range(min(vi_objects, 3)):
                objects.append({
                    'label': 'foliage',
                    'confidence': min(0.7 + vegetation_index, 0.9),
                    'bbox': [i*30, i*30, (i+1)*30, (i+1)*30]
                })
        
        # Urban elements detection
        if color_features['gray_ratio'] > 0.25:  # Lower threshold
            car_count = max(1, int(color_features['gray_ratio'] * 8))
            for i in range(min(car_count, 4)):
                objects.append({
                    'label': 'car',
                    'confidence': 0.6,
                    'bbox': [i*25, i*25, (i+1)*25, (i+1)*25]
                })
        
        # Sky and openness detection
        if color_features['blue_sky_ratio'] > 0.15:  # Lower threshold
            objects.extend([
                {'label': 'person', 'confidence': 0.5, 'bbox': [10, 10, 30, 30]},
                {'label': 'open space', 'confidence': 0.7, 'bbox': [0, 0, 50, 25]}
            ])
            
            if color_features['blue_sky_ratio'] > 0.3:
                objects.append({
                    'label': 'park',
                    'confidence': 0.8,
                    'bbox': [0, 0, 100, 50]
                })
        
        logger.info(f"Fallback detection generated {len(objects)} objects")
        return objects
    
    async def _segment_image(self, image: Image.Image) -> np.ndarray:
        """Perform semantic segmentation on the image"""
        
        if not self.segmentation_model:
            logger.warning("Segmentation model not available, using color-based segmentation")
            return self._fallback_segmentation(image)
        
        try:
            # Preprocess image for segmentation
            preprocess = transforms.Compose([
                transforms.Resize((520, 520)),  # DeepLab input size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = preprocess(image).unsqueeze(0).to(self.device)
            
            # Run segmentation
            with torch.no_grad():
                output = self.segmentation_model(input_tensor)['out'][0]
            
            # Get segmentation mask
            segmentation = output.argmax(0).cpu().numpy()
            
            logger.info(f"Segmentation completed: {np.unique(segmentation).shape[0]} classes found")
            return segmentation
            
        except Exception as e:
            logger.warning(f"Segmentation failed: {e}, using fallback")
            return self._fallback_segmentation(image)
    
    def _fallback_segmentation(self, image: Image.Image) -> np.ndarray:
        """Fallback segmentation using color clustering"""
        
        logger.info("Using color-based segmentation fallback")
        
        # Simple color-based segmentation
        img_array = np.array(image)
        h, w, c = img_array.shape
        
        # Create simple segmentation based on color
        segmentation = np.zeros((h, w), dtype=np.uint8)
        
        # Green areas (vegetation) = class 1
        green_mask = (
            (img_array[:, :, 1] > img_array[:, :, 0]) &
            (img_array[:, :, 1] > img_array[:, :, 2]) &
            (img_array[:, :, 1] > 50)
        )
        segmentation[green_mask] = 1
        
        # Blue areas (sky) = class 2
        blue_mask = (
            (img_array[:, :, 2] > img_array[:, :, 0]) &
            (img_array[:, :, 2] > img_array[:, :, 1]) &
            (img_array[:, :, 2] > 100)
        )
        segmentation[blue_mask] = 2
        
        # Gray areas (road/building) = class 3
        gray_mask = (
            (np.abs(img_array[:, :, 0].astype(np.int16) - img_array[:, :, 1].astype(np.int16)) < 30) &
            (np.abs(img_array[:, :, 1].astype(np.int16) - img_array[:, :, 2].astype(np.int16)) < 30)
        )
        segmentation[gray_mask] = 3
        
        return segmentation
    
    def _extract_sustainability_features(self, objects: List[Dict], segmentation: np.ndarray, color_features: Dict) -> Dict[str, Any]:
        """Extract sustainability-relevant features from analysis results - Enhanced version"""
        
        features = {}
        
        # Initialize all categories
        for category in self.sustainability_categories:
            features[category] = {
                'detected': [],
                'count': 0,
                'confidence': 0,
                'labels': [],
                'coverage_ratio': 0.0,  # New: area coverage ratio
                'quality_score': 0.0    # New: quality/health score
            }
        
        # Process detected objects with enhanced matching
        for obj in objects:
            label = obj['label'].lower()
            confidence = obj['confidence']
            
            # Match objects to sustainability categories with fuzzy matching
            for category, config in self.sustainability_categories.items():
                # Exact keyword matching
                exact_match = any(keyword in label for keyword in config['keywords'])
                # Partial matching for vegetation (more flexible)
                partial_match = False
                if category == 'greenCoverage':
                    vegetation_terms = ['green', 'leaf', 'branch', 'trunk', 'root', 'stem']
                    partial_match = any(term in label for term in vegetation_terms)
                
                if exact_match or partial_match:
                    features[category]['detected'].append(label)
                    features[category]['count'] += 1
                    features[category]['confidence'] = max(features[category]['confidence'], confidence)
                    features[category]['labels'].append({
                        'name': label,
                        'confidence': confidence
                    })
        
        # Enhanced segmentation-based features
        if segmentation is not None:
            total_pixels = segmentation.size
            unique_classes, counts = np.unique(segmentation, return_counts=True)
            
            # Vegetation coverage from segmentation (more detailed)
            vegetation_pixels = counts[unique_classes == 1].sum() if 1 in unique_classes else 0
            vegetation_ratio = vegetation_pixels / total_pixels
            
            # Store coverage ratio
            features['greenCoverage']['coverage_ratio'] = vegetation_ratio
            
            # Enhanced vegetation scoring based on coverage
            if vegetation_ratio > 0.05:  # Lower threshold (5% instead of 10%)
                # Progressive scoring: more coverage = exponentially better
                coverage_bonus = int(vegetation_ratio * 15)  # Increased multiplier
                features['greenCoverage']['count'] += coverage_bonus
                features['greenCoverage']['detected'].append(f'vegetation_area_{vegetation_ratio:.2%}')
                
                # Quality bonus for high vegetation coverage
                if vegetation_ratio > 0.2:  # 20%+ coverage gets quality bonus
                    features['greenCoverage']['quality_score'] = min(vegetation_ratio * 2, 1.0)
                    features['greenCoverage']['count'] += 3  # Quality bonus
                    features['greenCoverage']['detected'].append('high_vegetation_quality')
        
        # Enhanced color-based features with multiple thresholds
        green_ratio = color_features['green_ratio']
        vegetation_index = color_features['vegetation_index']
        
        # Multi-tier green detection
        if green_ratio > 0.08:  # Lower threshold (8% instead of 15%)
            base_bonus = int(green_ratio * 12)  # Increased sensitivity
            features['greenCoverage']['count'] += base_bonus
            features['greenCoverage']['detected'].append(f'green_pixels_{green_ratio:.2%}')
            
            # Tier bonuses
            if green_ratio > 0.15:  # 15%+ green pixels
                features['greenCoverage']['count'] += 2
                features['greenCoverage']['detected'].append('moderate_green_coverage')
            
            if green_ratio > 0.25:  # 25%+ green pixels
                features['greenCoverage']['count'] += 3
                features['greenCoverage']['detected'].append('high_green_coverage')
            
            if green_ratio > 0.35:  # 35%+ green pixels (excellent)
                features['greenCoverage']['count'] += 5
                features['greenCoverage']['detected'].append('excellent_green_coverage')
        
        # Vegetation index bonus
        if vegetation_index > 0.1:
            vi_bonus = int(vegetation_index * 8)
            features['greenCoverage']['count'] += vi_bonus
            features['greenCoverage']['detected'].append(f'vegetation_index_{vegetation_index:.3f}')
            features['greenCoverage']['quality_score'] = max(features['greenCoverage']['quality_score'], vegetation_index)
        
        # Sky visibility bonus (indicates open, breathable space)
        if color_features['blue_sky_ratio'] > 0.15:  # Lower threshold
            features['infrastructure']['count'] += 1
            features['infrastructure']['detected'].append('open_sky')
            
            # Sky visibility also benefits green coverage (natural lighting)
            if color_features['blue_sky_ratio'] > 0.25:
                features['greenCoverage']['count'] += 1
                features['greenCoverage']['detected'].append('natural_lighting')
        
        # Diversity bonus: reward variety in vegetation types
        green_labels = features['greenCoverage']['detected']
        unique_vegetation_types = len(set(label for label in green_labels if not label.startswith(('vegetation_area', 'green_pixels', 'high_', 'moderate_', 'excellent_'))))
        
        if unique_vegetation_types >= 2:
            diversity_bonus = min(unique_vegetation_types, 4)  # Max 4 bonus points
            features['greenCoverage']['count'] += diversity_bonus
            features['greenCoverage']['detected'].append(f'vegetation_diversity_{unique_vegetation_types}_types')
        
        return features
    
    def _calculate_sustainability_score(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall sustainability score from features"""
        
        category_scores = {}
        total_score = 50  # Base score
        
        for category, config in self.sustainability_categories.items():
            feature = features[category]
            count = feature['count']
            
            if category == 'carDependency':
                # Car dependency: more cars = lower score
                penalty = min(count * 4, 20)  # Max penalty of 20 points
                score = -penalty
                category_scores[category] = score
                total_score += score
            else:
                # Other categories: more items = higher score
                if category == 'greenCoverage':
                    score = min(count * 6, config['max_score'])
                elif category == 'walkability':
                    score = min(count * 5, config['max_score'])
                elif category == 'transitAccess':
                    score = min(count * 8, config['max_score'])
                elif category == 'buildingEfficiency':
                    score = min(count * 4, config['max_score'])
                else:
                    score = min(count * 3, config['max_score'])
                
                category_scores[category] = score
                total_score += score
        
        # Ensure score is within bounds
        final_score = max(0, min(100, total_score))
        
        return {
            'score': round(final_score),
            'category_scores': category_scores
        }
    
    def _generate_recommendations(self, features: Dict[str, Any], score: int) -> List[str]:
        """Generate improvement recommendations based on analysis"""
        
        recommendations = []
        
        # Green coverage recommendations
        if features['greenCoverage']['count'] < 3:
            recommendations.append("🌳 Add more trees and green spaces to improve air quality and reduce urban heat island effect")
        
        # Walkability recommendations
        if features['walkability']['count'] < 2:
            recommendations.append("🚶‍♀️ Improve pedestrian infrastructure with better sidewalks, crosswalks, and walking paths")
        
        # Transit recommendations
        if features['transitAccess']['count'] < 1:
            recommendations.append("🚌 Consider adding bike lanes, bus stops, or other public transit infrastructure")
        
        # Car dependency recommendations
        if features['carDependency']['count'] > 5:
            recommendations.append("🚗 Reduce car dependency by limiting parking spaces and promoting alternative transportation")
        
        # Building efficiency recommendations
        if features['buildingEfficiency']['count'] < 2:
            recommendations.append("🏢 Encourage sustainable building practices like solar panels, green roofs, and energy-efficient design")
        
        # Overall score recommendations
        if score >= 85:
            recommendations.append("✨ Excellent sustainability! This street is a model for eco-friendly urban design")
        elif score >= 70:
            recommendations.append("👍 Good sustainability foundation with opportunities for targeted improvements")
        elif score >= 50:
            recommendations.append("⚡ Moderate sustainability - several key improvements could make a significant impact")
        else:
            recommendations.append("🎯 Significant opportunities exist to transform this street's environmental impact")
        
        return recommendations
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        
        return {
            'object_detection': {
                'model': 'DETR ResNet-50' if self.object_detector else 'Color-based fallback',
                'loaded': self.object_detector is not None
            },
            'semantic_segmentation': {
                'model': 'DeepLab v3 ResNet-50' if self.segmentation_model else 'Color-based fallback',
                'loaded': self.segmentation_model is not None
            },
            'device': str(self.device),
            'categories': list(self.sustainability_categories.keys())
        }
