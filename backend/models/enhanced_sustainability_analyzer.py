"""
Enhanced Sustainability Analyzer with YOLOv8 and improved scoring
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import DetrImageProcessor, DetrForObjectDetection
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
import asyncio

from services.image_processor import ImageProcessor
from utils.logger import setup_logger

logger = setup_logger(__name__)

class EnhancedSustainabilityAnalyzer:
    """Enhanced analyzer with YOLOv8 and urban-specific scoring"""
    
    def __init__(self):
        """Initialize the enhanced sustainability analyzer"""
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Model components
        self.yolo_model = None
        self.detr_processor = None
        self.detr_model = None
        self.segmentation_model = None
        self.image_processor = ImageProcessor()
        
        # Enhanced sustainability categories with urban planning weights
        self.sustainability_categories = {
            'greenCoverage': {
                'weight': 0.30,  # Increased weight for environmental impact
                'keywords': ['tree', 'plant', 'grass', 'vegetation', 'bush', 'flower', 'garden', 'park'],
                'yolo_classes': [0],  # person (for parks with people)
                'max_score': 35,
                'urban_multiplier': 1.5  # Higher value in urban contexts
            },
            'walkability': {
                'weight': 0.25,  # High weight for livability
                'keywords': ['person', 'pedestrian', 'sidewalk', 'crosswalk', 'walkway'],
                'yolo_classes': [0],  # person
                'max_score': 30,
                'urban_multiplier': 1.3
            },
            'transitAccess': {
                'weight': 0.20,  # Important for sustainability
                'keywords': ['bus', 'bicycle', 'bike', 'train', 'subway', 'transit'],
                'yolo_classes': [1, 5],  # bicycle, bus
                'max_score': 25,
                'urban_multiplier': 1.4
            },
            'carDependency': {
                'weight': -0.20,  # Negative impact, increased penalty
                'keywords': ['car', 'truck', 'vehicle', 'parking', 'traffic'],
                'yolo_classes': [2, 3, 5, 7],  # car, motorcycle, bus, truck
                'max_score': 0,
                'penalty': -25,
                'urban_multiplier': 1.2
            },
            'buildingEfficiency': {
                'weight': 0.15,
                'keywords': ['solar panel', 'green roof', 'modern building'],
                'yolo_classes': [],  # Detected through segmentation
                'max_score': 20,
                'urban_multiplier': 1.1
            },
            'infrastructure': {
                'weight': 0.10,
                'keywords': ['traffic light', 'street light', 'bench', 'sign'],
                'yolo_classes': [9, 11, 12, 13],  # traffic light, stop sign, parking meter, bench
                'max_score': 15,
                'urban_multiplier': 1.0
            }
        }
        
        # YOLO COCO class names (relevant ones)
        self.yolo_class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench'
        }
    
    async def load_models(self):
        """Load all AI models with fallback handling"""
        
        logger.info("Loading enhanced AI models...")
        
        # Load YOLOv8 model (primary object detection)
        try:
            logger.info("Loading YOLOv8 model...")
            self.yolo_model = YOLO('yolov8n.pt')  # Nano version for speed
            logger.info("✅ YOLOv8 model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not load YOLOv8: {e}")
            self.yolo_model = None
        
        # Load DETR model (backup object detection)
        try:
            logger.info("Loading DETR model...")
            self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self.detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            self.detr_model.to(self.device)
            self.detr_model.eval()
            logger.info("✅ DETR model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not load DETR model: {e}")
            self.detr_model = None
        
        # Load segmentation model
        try:
            logger.info("Loading semantic segmentation model...")
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
        
        logger.info("🚀 Enhanced models loaded successfully!")
    
    async def analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Enhanced image analysis with multiple models
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with enhanced sustainability analysis
        """
        
        start_time = time.time()
        logger.info("Starting enhanced sustainability analysis...")
        
        try:
            # 1. YOLOv8 Object Detection (primary)
            logger.info("Step 1: YOLOv8 object detection...")
            yolo_objects = await self._detect_objects_yolo(image)
            
            # 2. DETR Object Detection (backup/additional)
            logger.info("Step 2: DETR object detection...")
            detr_objects = await self._detect_objects_detr(image)
            
            # 3. Semantic Segmentation
            logger.info("Step 3: Semantic segmentation...")
            segmentation = await self._segment_image(image)
            
            # 4. Enhanced color and texture analysis
            logger.info("Step 4: Enhanced feature extraction...")
            color_features = self.image_processor.extract_color_features(image)
            
            # 5. Combine all detection results
            combined_objects = self._combine_detections(yolo_objects, detr_objects)
            
            # 6. Extract enhanced sustainability features
            logger.info("Step 5: Enhanced sustainability feature extraction...")
            features = self._extract_enhanced_features(combined_objects, segmentation, color_features)
            
            # 7. Calculate enhanced sustainability score
            logger.info("Step 6: Enhanced scoring calculation...")
            score_result = self._calculate_enhanced_score(features)
            
            # 8. Generate intelligent recommendations
            recommendations = self._generate_intelligent_recommendations(features, score_result['score'])
            
            processing_time = time.time() - start_time
            
            result = {
                'score': score_result['score'],
                'method': 'enhanced_cnn_analysis',
                'features': features,
                'categoryScores': score_result['category_scores'],
                'recommendations': recommendations,
                'objects': combined_objects,
                'color_features': color_features,
                'model_performance': {
                    'yolo_objects': len(yolo_objects),
                    'detr_objects': len(detr_objects),
                    'total_objects': len(combined_objects),
                    'segmentation_classes': len(np.unique(segmentation)) if segmentation is not None else 0
                },
                'processing_time': round(processing_time, 2),
                'timestamp': datetime.now().isoformat(),
                'model_info': self.get_model_info()
            }
            
            logger.info(f"✅ Enhanced analysis complete! Score: {result['score']}/100 (took {processing_time:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Enhanced analysis failed: {str(e)}")
            raise e
    
    async def _detect_objects_yolo(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects using YOLOv8"""
        
        if not self.yolo_model:
            return []
        
        try:
            # Run YOLOv8 inference
            results = self.yolo_model(image, verbose=False)
            
            objects = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Stricter confidence thresholds to reduce false positives
                        min_confidence = 0.5  # Increased from 0.3
                        if class_id == 0:  # Person detection - be extra strict
                            min_confidence = 0.7
                        elif class_id in [2, 3, 5, 7]:  # Vehicles - moderately strict
                            min_confidence = 0.6
                        
                        if confidence > min_confidence:
                            objects.append({
                                'label': self.yolo_class_names.get(class_id, f'class_{class_id}'),
                                'confidence': confidence,
                                'bbox': box.xyxy[0].tolist(),
                                'source': 'yolo'
                            })
            
            logger.info(f"YOLOv8 detected {len(objects)} objects")
            return objects
            
        except Exception as e:
            logger.warning(f"YOLOv8 detection failed: {e}")
            return []
    
    async def _detect_objects_detr(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects using DETR (backup)"""
        
        if not self.detr_model or not self.detr_processor:
            return []
        
        try:
            # Preprocess image for DETR
            inputs = self.detr_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.detr_model(**inputs)
            
            # Process results with stricter thresholds
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.detr_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.5  # Increased from 0.3
            )[0]
            
            objects = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                label_name = self.detr_model.config.id2label[label.item()].lower()
                
                # Apply stricter thresholds based on object type
                min_confidence = 0.5
                if 'person' in label_name:
                    min_confidence = 0.7  # Extra strict for person detection
                elif any(vehicle in label_name for vehicle in ['car', 'truck', 'bus', 'motorcycle']):
                    min_confidence = 0.6  # Moderately strict for vehicles
                
                if score > min_confidence:
                    objects.append({
                        'label': self.detr_model.config.id2label[label.item()],
                        'confidence': score.item(),
                        'bbox': box.tolist(),
                        'source': 'detr'
                    })
            
            logger.info(f"DETR detected {len(objects)} objects")
            return objects
            
        except Exception as e:
            logger.warning(f"DETR detection failed: {e}")
            return []
    
    def _combine_detections(self, yolo_objects: List[Dict], detr_objects: List[Dict]) -> List[Dict]:
        """Combine and deduplicate detections from multiple models"""
        
        # Start with YOLOv8 results (generally more accurate)
        combined = yolo_objects.copy()
        
        # Add DETR results that don't overlap significantly
        for detr_obj in detr_objects:
            # Simple deduplication based on label and confidence
            is_duplicate = False
            for yolo_obj in yolo_objects:
                if (detr_obj['label'].lower() == yolo_obj['label'].lower() and 
                    abs(detr_obj['confidence'] - yolo_obj['confidence']) < 0.2):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                combined.append(detr_obj)
        
        # Sort by confidence
        combined.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"Combined detections: {len(combined)} total objects")
        return combined
    
    async def _segment_image(self, image: Image.Image) -> np.ndarray:
        """Perform semantic segmentation"""
        
        if not self.segmentation_model:
            logger.warning("Segmentation model not available, using color-based segmentation")
            return self._fallback_segmentation(image)
        
        try:
            # Preprocess image for segmentation
            preprocess = transforms.Compose([
                transforms.Resize((520, 520)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = preprocess(image).unsqueeze(0).to(self.device)
            
            # Run segmentation
            with torch.no_grad():
                output = self.segmentation_model(input_tensor)['out'][0]
            
            segmentation = output.argmax(0).cpu().numpy()
            
            logger.info(f"Segmentation completed: {np.unique(segmentation).shape[0]} classes found")
            return segmentation
            
        except Exception as e:
            logger.warning(f"Segmentation failed: {e}, using fallback")
            return self._fallback_segmentation(image)
    
    def _fallback_segmentation(self, image: Image.Image) -> np.ndarray:
        """Fallback segmentation using enhanced color analysis"""
        
        img_array = np.array(image)
        h, w, c = img_array.shape
        segmentation = np.zeros((h, w), dtype=np.uint8)
        
        # Enhanced color-based segmentation
        # Green areas (vegetation) = class 1
        green_mask = (
            (img_array[:, :, 1] > img_array[:, :, 0] + 10) &
            (img_array[:, :, 1] > img_array[:, :, 2] + 5) &
            (img_array[:, :, 1] > 40)
        )
        segmentation[green_mask] = 1
        
        # Blue areas (sky/water) = class 2
        blue_mask = (
            (img_array[:, :, 2] > img_array[:, :, 0] + 15) &
            (img_array[:, :, 2] > img_array[:, :, 1] + 10) &
            (img_array[:, :, 2] > 80)
        )
        segmentation[blue_mask] = 2
        
        # Gray areas (road/concrete) = class 3
        gray_mask = (
            (np.abs(img_array[:, :, 0].astype(np.int16) - img_array[:, :, 1].astype(np.int16)) < 25) &
            (np.abs(img_array[:, :, 1].astype(np.int16) - img_array[:, :, 2].astype(np.int16)) < 25) &
            (img_array[:, :, 0] > 50) & (img_array[:, :, 0] < 200)
        )
        segmentation[gray_mask] = 3
        
        return segmentation
    
    def _extract_enhanced_features(self, objects: List[Dict], segmentation: np.ndarray, color_features: Dict) -> Dict[str, Any]:
        """Extract enhanced sustainability features"""
        
        features = {}
        
        # Initialize all categories
        for category in self.sustainability_categories:
            features[category] = {
                'detected': [],
                'count': 0,
                'confidence': 0,
                'labels': [],
                'yolo_count': 0,
                'detr_count': 0
            }
        
        # Process detected objects with enhanced logic
        for obj in objects:
            label = obj['label'].lower()
            confidence = obj['confidence']
            source = obj.get('source', 'unknown')
            
            # Match objects to sustainability categories
            for category, config in self.sustainability_categories.items():
                # Check keyword matches
                keyword_match = any(keyword in label for keyword in config['keywords'])
                
                # Check YOLO class matches
                yolo_class_match = False
                if source == 'yolo':
                    for class_id, class_name in self.yolo_class_names.items():
                        if class_name.lower() == label and class_id in config.get('yolo_classes', []):
                            yolo_class_match = True
                            break
                
                if keyword_match or yolo_class_match:
                    features[category]['detected'].append(label)
                    features[category]['count'] += 1
                    features[category]['confidence'] = max(features[category]['confidence'], confidence)
                    features[category]['labels'].append({
                        'name': label,
                        'confidence': confidence,
                        'source': source
                    })
                    
                    if source == 'yolo':
                        features[category]['yolo_count'] += 1
                    elif source == 'detr':
                        features[category]['detr_count'] += 1
        
        # Enhanced segmentation-based features
        if segmentation is not None:
            total_pixels = segmentation.size
            unique_classes, counts = np.unique(segmentation, return_counts=True)
            
            # Vegetation coverage (class 1)
            vegetation_pixels = counts[unique_classes == 1].sum() if 1 in unique_classes else 0
            vegetation_ratio = vegetation_pixels / total_pixels
            
            # Enhanced green coverage scoring
            if vegetation_ratio > 0.05:  # At least 5% vegetation
                bonus_count = int(vegetation_ratio * 15)  # More generous scoring
                features['greenCoverage']['count'] += bonus_count
                features['greenCoverage']['detected'].append(f'vegetation_area_{vegetation_ratio:.2f}')
        
        # Enhanced color-based features
        if color_features['green_ratio'] > 0.10:
            features['greenCoverage']['count'] += 3
            features['greenCoverage']['detected'].append('color_green_areas')
        
        if color_features['blue_sky_ratio'] > 0.15:
            features['infrastructure']['count'] += 2
            features['infrastructure']['detected'].append('open_sky_visibility')
        
        return features
    
    def _calculate_enhanced_score(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate enhanced sustainability score with urban planning principles"""
        
        category_scores = {}
        total_score = 60  # Higher base score for urban areas
        
        for category, config in self.sustainability_categories.items():
            feature = features[category]
            count = feature['count']
            confidence_bonus = feature['confidence'] * 0.1  # Confidence bonus
            
            if category == 'carDependency':
                # Enhanced car dependency penalty
                base_penalty = min(count * 3, 20)
                confidence_penalty = confidence_bonus * 2  # Higher penalty for confident car detections
                total_penalty = base_penalty + confidence_penalty
                score = -total_penalty
                category_scores[category] = score
                total_score += score
            else:
                # Enhanced positive scoring
                base_score = 0
                
                if category == 'greenCoverage':
                    # More generous green scoring
                    base_score = min(count * 4, config['max_score'])
                    # Bonus for diverse vegetation types
                    if len(set(feature['detected'])) > 2:
                        base_score += 5
                        
                elif category == 'walkability':
                    # Enhanced walkability scoring
                    base_score = min(count * 6, config['max_score'])
                    # Bonus for high confidence pedestrian detection
                    if feature['confidence'] > 0.8:
                        base_score += 3
                        
                elif category == 'transitAccess':
                    # Higher value for transit
                    base_score = min(count * 10, config['max_score'])
                    # Bonus for multiple transit types
                    if 'bicycle' in feature['detected'] and 'bus' in feature['detected']:
                        base_score += 5
                        
                elif category == 'buildingEfficiency':
                    base_score = min(count * 5, config['max_score'])
                    
                else:  # infrastructure
                    base_score = min(count * 4, config['max_score'])
                
                # Apply urban multiplier
                final_score = base_score * config.get('urban_multiplier', 1.0)
                final_score += confidence_bonus
                
                category_scores[category] = round(final_score)
                total_score += final_score
        
        # Ensure score is within bounds
        final_score = max(0, min(100, total_score))
        
        return {
            'score': round(final_score),
            'category_scores': category_scores
        }
    
    def _generate_intelligent_recommendations(self, features: Dict[str, Any], score: int) -> List[str]:
        """Generate intelligent recommendations based on detailed analysis"""
        
        recommendations = []
        
        # Analyze green coverage
        green_count = features['greenCoverage']['count']
        if green_count < 2:
            recommendations.append("🌳 Critical: Add street trees and green infrastructure - current vegetation is insufficient for urban sustainability")
        elif green_count < 5:
            recommendations.append("🌱 Moderate: Increase green coverage with additional trees, green walls, or pocket parks")
        else:
            recommendations.append("🌿 Excellent: Great green coverage! Consider diversifying plant species for biodiversity")
        
        # Analyze walkability
        walk_count = features['walkability']['count']
        if walk_count < 1:
            recommendations.append("🚶‍♀️ Critical: No pedestrian activity detected - improve sidewalk quality and pedestrian safety")
        elif walk_count < 3:
            recommendations.append("👥 Moderate: Some pedestrian activity - enhance crosswalks and pedestrian amenities")
        else:
            recommendations.append("🚶‍♂️ Good: Active pedestrian environment - maintain and expand walkable infrastructure")
        
        # Analyze transit access
        transit_count = features['transitAccess']['count']
        if transit_count < 1:
            recommendations.append("🚌 Critical: No public transit detected - consider bus routes, bike lanes, or micro-mobility options")
        else:
            recommendations.append("🚲 Good: Transit options available - consider expanding multi-modal transportation")
        
        # Analyze car dependency
        car_count = features['carDependency']['count']
        if car_count > 8:
            recommendations.append("🚗 High car dependency detected - implement car-free zones and promote alternative transport")
        elif car_count > 4:
            recommendations.append("🚙 Moderate car presence - balance parking with pedestrian and cycling infrastructure")
        else:
            recommendations.append("🚶‍♀️ Low car dependency - excellent for sustainable urban mobility")
        
        # Overall score-based recommendations
        if score >= 85:
            recommendations.append("⭐ Outstanding sustainability score! This street exemplifies excellent urban planning")
        elif score >= 70:
            recommendations.append("✅ Good sustainability foundation - focus on targeted improvements for maximum impact")
        elif score >= 50:
            recommendations.append("📈 Moderate sustainability - strategic interventions can significantly improve livability")
        else:
            recommendations.append("🎯 Major sustainability improvements needed - prioritize green infrastructure and transit access")
        
        return recommendations
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded enhanced models"""
        
        return {
            'object_detection': {
                'primary': 'YOLOv8 Nano' if self.yolo_model else 'Not loaded',
                'backup': 'DETR ResNet-50' if self.detr_model else 'Not loaded',
                'yolo_loaded': self.yolo_model is not None,
                'detr_loaded': self.detr_model is not None
            },
            'semantic_segmentation': {
                'model': 'DeepLab v3 ResNet-50' if self.segmentation_model else 'Color-based fallback',
                'loaded': self.segmentation_model is not None
            },
            'device': str(self.device),
            'categories': list(self.sustainability_categories.keys()),
            'enhancement_level': 'professional'
        }
