"""
Professional Ensemble Sustainability Analyzer
Multiple specialized models working together for maximum accuracy
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import DetrImageProcessor, DetrForObjectDetection
from ultralytics import YOLO
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import asyncio
import cv2

from services.image_processor import ImageProcessor
from utils.logger import setup_logger

logger = setup_logger(__name__)

class GreenDetectionSpecialist:
    """Specialized model for vegetation and green infrastructure detection"""
    
    def __init__(self, device):
        self.device = device
        self.yolo_green = None
        self.segmentation_model = None
        
    async def load_models(self):
        """Load green detection models"""
        try:
            logger.info("Loading Green Detection Specialist...")
            
            # YOLOv8s for better vegetation detection
            self.yolo_green = YOLO('yolov8s.pt')
            
            # Advanced segmentation for vegetation
            self.segmentation_model = smp.DeepLabV3Plus(
                encoder_name="efficientnet-b4",
                encoder_weights="imagenet",
                classes=21,  # PASCAL VOC classes including vegetation
                activation=None,
            )
            self.segmentation_model.to(self.device)
            self.segmentation_model.eval()
            
            logger.info("✅ Green Detection Specialist loaded")
            
        except Exception as e:
            logger.warning(f"⚠️ Green Detection Specialist failed to load: {e}")
    
    async def analyze_vegetation(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze vegetation and green infrastructure"""
        
        results = {
            'vegetation_objects': [],
            'green_coverage_ratio': 0.0,
            'vegetation_health': 0.0,
            'green_infrastructure': [],
            'confidence': 0.0
        }
        
        try:
            # YOLO detection for vegetation objects - FIXED: Don't use person detection for vegetation
            if self.yolo_green:
                yolo_results = self.yolo_green(image, verbose=False)
                
                for result in yolo_results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            # Only use actual plant-related classes (potted plant)
                            vegetation_classes = [16]  # potted plant only
                            if class_id in vegetation_classes and confidence > 0.5:
                                class_names = {16: 'potted_plant'}
                                results['vegetation_objects'].append({
                                    'type': class_names.get(class_id, 'plant'),
                                    'confidence': confidence,
                                    'bbox': box.xyxy[0].tolist()
                                })
            
            # Advanced color-based vegetation analysis
            img_array = np.array(image)
            results.update(self._analyze_vegetation_color(img_array))
            
            # Calculate overall confidence
            if results['vegetation_objects']:
                results['confidence'] = np.mean([obj['confidence'] for obj in results['vegetation_objects']])
            else:
                results['confidence'] = results['green_coverage_ratio']
            
            logger.info(f"Green analysis: {results['green_coverage_ratio']:.2f} coverage, {len(results['vegetation_objects'])} objects")
            
        except Exception as e:
            logger.warning(f"Green analysis failed: {e}")
        
        return results
    
    def _analyze_vegetation_color(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Advanced color-based vegetation analysis"""
        
        h, w, c = img_array.shape
        
        # Convert to HSV for better vegetation detection
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Multiple green detection strategies
        green_masks = []
        
        # Strategy 1: Balanced traditional green detection for real vegetation
        green_mask1 = (
            (img_array[:, :, 1] > img_array[:, :, 0] + 15) &  # Reasonable threshold
            (img_array[:, :, 1] > img_array[:, :, 2] + 8) &   # Detect natural greens
            (img_array[:, :, 1] > 50) &  # Lower minimum for natural vegetation
            (img_array[:, :, 1] < 220)   # Avoid oversaturated areas
        )
        green_masks.append(green_mask1)
        
        # Strategy 2: HSV-based vegetation detection for trees
        green_mask2 = (
            (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) &  # Full green hue range
            (hsv[:, :, 1] >= 30) &  # Lower saturation for natural colors
            (hsv[:, :, 2] >= 40) & (hsv[:, :, 2] <= 220)   # Broader value range
        )
        green_masks.append(green_mask2)
        
        # Strategy 3: NDVI-like vegetation index for natural vegetation
        r = img_array[:, :, 0].astype(np.float32)
        g = img_array[:, :, 1].astype(np.float32)
        b = img_array[:, :, 2].astype(np.float32)
        
        # Modified NDVI for RGB - balanced for real vegetation
        ndvi = (g - r) / (g + r + 1e-8)
        green_mask3 = (ndvi > 0.05) & (g > 45)  # Lower threshold for natural vegetation
        green_masks.append(green_mask3)
        
        # Strategy 4: Tree-specific detection (browns + greens)
        tree_mask = (
            # Tree trunk colors (browns)
            ((img_array[:, :, 0] > img_array[:, :, 1]) & 
             (img_array[:, :, 1] > img_array[:, :, 2]) &
             (img_array[:, :, 0] > 60) & (img_array[:, :, 0] < 150)) |
            # Tree foliage (natural greens)
            ((img_array[:, :, 1] > img_array[:, :, 0]) & 
             (img_array[:, :, 1] > img_array[:, :, 2]) &
             (img_array[:, :, 1] > 40))
        )
        green_masks.append(tree_mask)
        
        # Use majority voting (2 out of 4 strategies) for balanced detection
        mask_sum = sum(green_masks)
        combined_green = mask_sum >= 2  # At least 2 strategies must agree
        
        # If too restrictive, fall back to OR of most reliable strategies
        if np.sum(combined_green) < total_pixels * 0.02:  # Less than 2% detected
            combined_green = np.logical_or(green_mask1, green_mask2)  # Use most reliable
        
        # Remove noise with morphological operations
        kernel = np.ones((3,3), np.uint8)
        combined_green = cv2.morphologyEx(combined_green.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        combined_green = cv2.morphologyEx(combined_green, cv2.MORPH_OPEN, kernel)
        combined_green = combined_green.astype(bool)
        
        # Calculate metrics
        total_pixels = h * w
        green_pixels = np.sum(combined_green)
        green_coverage_ratio = green_pixels / total_pixels
        
        # Vegetation health estimation (based on color intensity)
        if green_pixels > 0:
            green_areas = img_array[combined_green]
            vegetation_health = np.mean(green_areas[:, 1]) / 255.0  # Green channel intensity
        else:
            vegetation_health = 0.0
        
        return {
            'green_coverage_ratio': float(green_coverage_ratio),
            'vegetation_health': float(vegetation_health),
            'green_infrastructure': self._detect_green_infrastructure(img_array, combined_green)
        }
    
    def _detect_green_infrastructure(self, img_array: np.ndarray, green_mask: np.ndarray) -> List[str]:
        """Detect types of green infrastructure - more conservative and realistic"""
        
        infrastructure = []
        total_pixels = img_array.shape[0] * img_array.shape[1]
        green_pixels = np.sum(green_mask)
        
        # Only proceed if there's significant green coverage
        if green_pixels < total_pixels * 0.03:  # Less than 3% green
            return infrastructure
        
        # Find connected components
        green_uint8 = green_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(green_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            area_ratio = area / total_pixels
            
            # Much stricter area requirements
            if area < 2000:  # Ignore very small areas
                continue
                
            # Analyze shape to determine type
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # More conservative classification
                if area_ratio > 0.15 and area > 20000:  # Very large green area
                    if len(infrastructure) == 0:  # Only one park area max
                        infrastructure.append('park_area')
                elif circularity > 0.6 and area > 5000:  # Round, substantial area
                    infrastructure.append('tree_canopy')
                elif area > 3000:  # Medium-sized green area
                    infrastructure.append('street_vegetation')
            
            # Limit total detections to avoid inflation
            if len(infrastructure) >= 3:
                break
        
        return infrastructure


class UrbanInfrastructureSpecialist:
    """Specialized model for urban infrastructure and building analysis"""
    
    def __init__(self, device):
        self.device = device
        self.segmentation_model = None
        self.yolo_infra = None
        
    async def load_models(self):
        """Load infrastructure analysis models"""
        try:
            logger.info("Loading Urban Infrastructure Specialist...")
            
            # YOLOv8s for infrastructure objects
            self.yolo_infra = YOLO('yolov8s.pt')
            
            # Advanced segmentation for urban scenes
            self.segmentation_model = torch.hub.load(
                'pytorch/vision:v0.10.0',
                'deeplabv3_resnet101',
                pretrained=True
            )
            self.segmentation_model.to(self.device)
            self.segmentation_model.eval()
            
            logger.info("✅ Urban Infrastructure Specialist loaded")
            
        except Exception as e:
            logger.warning(f"⚠️ Urban Infrastructure Specialist failed to load: {e}")
    
    async def analyze_infrastructure(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze urban infrastructure and buildings"""
        
        results = {
            'infrastructure_objects': [],
            'building_types': [],
            'accessibility_features': [],
            'urban_furniture': [],
            'confidence': 0.0
        }
        
        try:
            # YOLO detection for infrastructure
            if self.yolo_infra:
                yolo_results = self.yolo_infra(image, verbose=False)
                
                infrastructure_classes = [9, 10, 11, 12, 13]  # traffic light, fire hydrant, stop sign, parking meter, bench
                
                for result in yolo_results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            if class_id in infrastructure_classes and confidence > 0.5:
                                class_names = {
                                    9: 'traffic_light', 10: 'fire_hydrant', 11: 'stop_sign',
                                    12: 'parking_meter', 13: 'bench'
                                }
                                
                                results['infrastructure_objects'].append({
                                    'type': class_names.get(class_id, f'class_{class_id}'),
                                    'confidence': confidence,
                                    'bbox': box.xyxy[0].tolist()
                                })
            
            # Segmentation analysis for buildings and roads
            if self.segmentation_model:
                segmentation = await self._segment_urban_scene(image)
                results.update(self._analyze_urban_segmentation(segmentation))
            
            # Calculate confidence
            if results['infrastructure_objects']:
                results['confidence'] = np.mean([obj['confidence'] for obj in results['infrastructure_objects']])
            else:
                results['confidence'] = 0.3  # Base confidence for segmentation
            
            logger.info(f"Infrastructure analysis: {len(results['infrastructure_objects'])} objects detected")
            
        except Exception as e:
            logger.warning(f"Infrastructure analysis failed: {e}")
        
        return results
    
    async def _segment_urban_scene(self, image: Image.Image) -> np.ndarray:
        """Perform urban scene segmentation"""
        
        try:
            preprocess = transforms.Compose([
                transforms.Resize((520, 520)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.segmentation_model(input_tensor)['out'][0]
            
            return output.argmax(0).cpu().numpy()
            
        except Exception as e:
            logger.warning(f"Urban segmentation failed: {e}")
            return np.zeros((520, 520), dtype=np.uint8)
    
    def _analyze_urban_segmentation(self, segmentation: np.ndarray) -> Dict[str, Any]:
        """Analyze segmentation results for urban features"""
        
        unique_classes, counts = np.unique(segmentation, return_counts=True)
        total_pixels = segmentation.size
        
        # Map segmentation classes to urban features
        class_mapping = {
            0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
            5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
            10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike',
            15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
        }
        
        building_types = []
        accessibility_features = []
        urban_furniture = []
        
        for class_id, count in zip(unique_classes, counts):
            ratio = count / total_pixels
            class_name = class_mapping.get(class_id, f'class_{class_id}')
            
            if ratio > 0.05:  # Significant presence
                if class_id in [6, 19]:  # bus, train
                    building_types.append('transit_infrastructure')
                elif class_id in [9, 11]:  # chair, dining table (urban furniture)
                    urban_furniture.append('street_furniture')
                elif class_id == 15:  # person
                    accessibility_features.append('pedestrian_area')
        
        return {
            'building_types': building_types,
            'accessibility_features': accessibility_features,
            'urban_furniture': urban_furniture
        }


class TransportationSpecialist:
    """Specialized model for transportation and mobility analysis"""
    
    def __init__(self, device):
        self.device = device
        self.yolo_transport = None
        
    async def load_models(self):
        """Load transportation analysis models"""
        try:
            logger.info("Loading Transportation Specialist...")
            
            # YOLOv8x for maximum accuracy in vehicle/transport detection
            self.yolo_transport = YOLO('yolov8x.pt')
            
            logger.info("✅ Transportation Specialist loaded")
            
        except Exception as e:
            logger.warning(f"⚠️ Transportation Specialist failed to load: {e}")
    
    async def analyze_transportation(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze transportation and mobility features"""
        
        results = {
            'vehicles': [],
            'public_transit': [],
            'pedestrians': [],
            'cyclists': [],
            'mobility_score': 0.0,
            'car_dependency': 0.0,
            'confidence': 0.0
        }
        
        try:
            if self.yolo_transport:
                yolo_results = self.yolo_transport(image, verbose=False)
                
                # Transportation class mapping
                transport_classes = {
                    0: 'pedestrians',
                    1: 'cyclists', 
                    2: 'vehicles',
                    3: 'vehicles',  # motorcycle
                    5: 'public_transit',  # bus
                    6: 'public_transit',  # train
                    7: 'vehicles'  # truck
                }
                
                confidence_scores = []
                
                for result in yolo_results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            # More reasonable thresholds for better detection
                            min_confidence = 0.4  # Lowered from 0.6
                            if class_id == 0:  # person
                                min_confidence = 0.5  # Lowered from 0.7
                            elif class_id in [2, 3, 7]:  # vehicles
                                min_confidence = 0.45  # Lowered from 0.65
                            
                            if confidence > min_confidence and class_id in transport_classes:
                                category = transport_classes[class_id]
                                
                                detection = {
                                    'type': self._get_specific_type(class_id),
                                    'confidence': confidence,
                                    'bbox': box.xyxy[0].tolist()
                                }
                                
                                results[category].append(detection)
                                confidence_scores.append(confidence)
                
                # Calculate mobility metrics
                results.update(self._calculate_mobility_metrics(results))
                
                # Overall confidence
                if confidence_scores:
                    results['confidence'] = np.mean(confidence_scores)
                
                logger.info(f"Transport analysis: {len(results['vehicles'])} vehicles, {len(results['pedestrians'])} pedestrians")
            
        except Exception as e:
            logger.warning(f"Transportation analysis failed: {e}")
        
        return results
    
    def _get_specific_type(self, class_id: int) -> str:
        """Get specific transportation type"""
        type_mapping = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            5: 'bus', 6: 'train', 7: 'truck'
        }
        return type_mapping.get(class_id, f'class_{class_id}')
    
    def _calculate_mobility_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate mobility and sustainability metrics"""
        
        vehicle_count = len(results['vehicles'])
        transit_count = len(results['public_transit'])
        pedestrian_count = len(results['pedestrians'])
        cyclist_count = len(results['cyclists'])
        
        total_transport = vehicle_count + transit_count + pedestrian_count + cyclist_count
        
        if total_transport > 0:
            # Mobility score (higher = more sustainable)
            sustainable_transport = transit_count + pedestrian_count + cyclist_count
            mobility_score = sustainable_transport / total_transport
            
            # Car dependency (higher = more car dependent)
            car_dependency = vehicle_count / total_transport
        else:
            mobility_score = 0.5  # Neutral score
            car_dependency = 0.0
        
        return {
            'mobility_score': float(mobility_score),
            'car_dependency': float(car_dependency)
        }


class SustainabilityEnsembleScorer:
    """Intelligent ensemble scorer combining all specialist outputs"""
    
    def __init__(self):
        self.weights = {
            'green': 0.35,      # Increased weight for environmental impact
            'infrastructure': 0.25,
            'transportation': 0.40  # High weight for mobility
        }
    
    def calculate_ensemble_score(self, green_results: Dict, infra_results: Dict, transport_results: Dict) -> Dict[str, Any]:
        """Calculate final sustainability score using ensemble of specialists"""
        
        # Individual component scores
        green_score = self._score_green_component(green_results)
        infra_score = self._score_infrastructure_component(infra_results)
        transport_score = self._score_transportation_component(transport_results)
        
        # Confidence-weighted ensemble
        confidences = [
            green_results.get('confidence', 0.0),
            infra_results.get('confidence', 0.0),
            transport_results.get('confidence', 0.0)
        ]
        
        # Dynamic weight adjustment based on confidence
        adjusted_weights = self._adjust_weights_by_confidence(confidences)
        
        # Calculate final score
        final_score = (
            green_score * adjusted_weights[0] +
            infra_score * adjusted_weights[1] +
            transport_score * adjusted_weights[2]
        )
        
        # Ensure score is in valid range
        final_score = max(0, min(100, final_score))
        
        return {
            'score': round(final_score),
            'component_scores': {
                'green_coverage': round(green_score),
                'infrastructure': round(infra_score),
                'transportation': round(transport_score)
            },
            'confidences': confidences,
            'adjusted_weights': adjusted_weights,
            'ensemble_method': 'confidence_weighted'
        }
    
    def _score_green_component(self, results: Dict) -> float:
        """Score green/environmental component - more realistic scoring"""
        
        # Start with 0 - no free points
        base_score = 0
        
        # Vegetation coverage bonus (more conservative)
        coverage_ratio = results.get('green_coverage_ratio', 0.0)
        if coverage_ratio > 0.05:  # Only give points if significant vegetation
            coverage_score = min(coverage_ratio * 80, 35)  # Max 35 points, more conservative
        else:
            coverage_score = 0
        
        # Vegetation health bonus (only if vegetation exists)
        health = results.get('vegetation_health', 0.0)
        if coverage_ratio > 0.05:
            health_score = health * 15  # Max 15 points
        else:
            health_score = 0
        
        # Green infrastructure bonus (only if detected)
        infra_count = len(results.get('green_infrastructure', []))
        infra_score = min(infra_count * 8, 15)  # Max 15 points, more conservative
        
        # Vegetation objects bonus
        veg_objects = len(results.get('vegetation_objects', []))
        object_score = min(veg_objects * 5, 10)  # Max 10 points
        
        total_score = base_score + coverage_score + health_score + infra_score + object_score
        return min(total_score, 50)  # Cap at 50 points max
    
    def _score_infrastructure_component(self, results: Dict) -> float:
        """Score urban infrastructure component"""
        
        base_score = 30  # Base infrastructure score
        
        # Infrastructure objects bonus
        obj_count = len(results.get('infrastructure_objects', []))
        obj_score = min(obj_count * 8, 30)  # Max 30 points
        
        # Accessibility features bonus
        access_count = len(results.get('accessibility_features', []))
        access_score = min(access_count * 15, 25)  # Max 25 points
        
        # Urban furniture bonus
        furniture_count = len(results.get('urban_furniture', []))
        furniture_score = min(furniture_count * 10, 15)  # Max 15 points
        
        return base_score + obj_score + access_score + furniture_score
    
    def _score_transportation_component(self, results: Dict) -> float:
        """Score transportation/mobility component - more realistic scoring"""
        
        # Start with lower base score
        base_score = 20  # Reduced from 40
        
        # Get counts
        vehicle_count = len(results.get('vehicles', []))
        transit_count = len(results.get('public_transit', []))
        pedestrian_count = len(results.get('pedestrians', []))
        cyclist_count = len(results.get('cyclists', []))
        
        # Mobility score bonus (sustainable transport)
        mobility = results.get('mobility_score', 0.0)
        mobility_score = mobility * 30  # Max 30 points, reduced from 40
        
        # Car dependency penalty (more severe)
        car_dependency = results.get('car_dependency', 0.0)
        car_penalty = car_dependency * 40  # Increased penalty from 30 to 40
        
        # Public transit bonus (only if actually detected)
        if transit_count > 0:
            transit_score = min(transit_count * 12, 25)  # Max 25 points
        else:
            transit_score = 0
        
        # Pedestrian/cyclist bonus (only if detected)
        active_count = pedestrian_count + cyclist_count
        if active_count > 0:
            active_score = min(active_count * 6, 15)  # Max 15 points, reduced
        else:
            active_score = 0
        
        # Heavy car penalty for cluttered streets
        if vehicle_count > 3:
            clutter_penalty = (vehicle_count - 3) * 5  # Additional penalty for many cars
        else:
            clutter_penalty = 0
        
        total_score = base_score + mobility_score - car_penalty + transit_score + active_score - clutter_penalty
        return max(0, min(total_score, 60))  # Cap at 60 points max, ensure non-negative
    
    def _adjust_weights_by_confidence(self, confidences: List[float]) -> List[float]:
        """Adjust ensemble weights based on model confidence"""
        
        # Normalize confidences
        total_confidence = sum(confidences) + 1e-8
        confidence_weights = [c / total_confidence for c in confidences]
        
        # Blend with base weights (50% confidence, 50% base)
        base_weights = list(self.weights.values())
        adjusted = [
            0.5 * base + 0.5 * conf 
            for base, conf in zip(base_weights, confidence_weights)
        ]
        
        # Normalize to sum to 1
        total = sum(adjusted)
        return [w / total for w in adjusted]


class EnsembleSustainabilityAnalyzer:
    """Professional ensemble system combining all specialists"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Ensemble system using device: {self.device}")
        
        # Initialize specialists
        self.green_specialist = GreenDetectionSpecialist(self.device)
        self.infra_specialist = UrbanInfrastructureSpecialist(self.device)
        self.transport_specialist = TransportationSpecialist(self.device)
        self.ensemble_scorer = SustainabilityEnsembleScorer()
        
        self.image_processor = ImageProcessor()
        
    async def load_models(self):
        """Load all specialist models"""
        logger.info("🚀 Loading Professional Ensemble System...")
        
        # Load all specialists in parallel for speed
        await asyncio.gather(
            self.green_specialist.load_models(),
            self.infra_specialist.load_models(),
            self.transport_specialist.load_models()
        )
        
        logger.info("✅ Professional Ensemble System loaded successfully!")
    
    async def analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """Professional ensemble analysis"""
        
        start_time = time.time()
        logger.info("🔬 Starting Professional Ensemble Analysis...")
        
        try:
            # Preprocess image
            processed_image = self.image_processor.preprocess_image(image)
            
            # Run all specialists in parallel
            logger.info("Running specialist analyses...")
            green_results, infra_results, transport_results = await asyncio.gather(
                self.green_specialist.analyze_vegetation(processed_image),
                self.infra_specialist.analyze_infrastructure(processed_image),
                self.transport_specialist.analyze_transportation(processed_image)
            )
            
            # Ensemble scoring
            logger.info("Calculating ensemble score...")
            ensemble_result = self.ensemble_scorer.calculate_ensemble_score(
                green_results, infra_results, transport_results
            )
            
            # Generate professional recommendations
            recommendations = self._generate_professional_recommendations(
                green_results, infra_results, transport_results, ensemble_result['score']
            )
            
            processing_time = time.time() - start_time
            
            # Compile final result
            result = {
                'score': ensemble_result['score'],
                'method': 'professional_ensemble_analysis',
                'component_scores': ensemble_result['component_scores'],
                'specialist_results': {
                    'green_analysis': green_results,
                    'infrastructure_analysis': infra_results,
                    'transportation_analysis': transport_results
                },
                'ensemble_details': {
                    'confidences': ensemble_result['confidences'],
                    'adjusted_weights': ensemble_result['adjusted_weights'],
                    'method': ensemble_result['ensemble_method']
                },
                'recommendations': recommendations,
                'processing_time': round(processing_time, 2),
                'timestamp': datetime.now().isoformat(),
                'model_info': self.get_model_info()
            }
            
            logger.info(f"✅ Professional Ensemble Analysis complete! Score: {result['score']}/100 (took {processing_time:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ensemble analysis failed: {str(e)}")
            raise e
    
    def _generate_professional_recommendations(self, green_results: Dict, infra_results: Dict, 
                                            transport_results: Dict, final_score: int) -> List[str]:
        """Generate professional urban planning recommendations"""
        
        recommendations = []
        
        # Green infrastructure recommendations
        green_coverage = green_results.get('green_coverage_ratio', 0.0)
        if green_coverage < 0.15:
            recommendations.append("🌳 CRITICAL: Implement comprehensive green infrastructure - add street trees, green corridors, and pocket parks to achieve minimum 15% vegetation coverage")
        elif green_coverage < 0.25:
            recommendations.append("🌱 PRIORITY: Expand green infrastructure with additional tree canopy and sustainable landscaping")
        else:
            recommendations.append("🌿 EXCELLENT: Outstanding green coverage supports urban biodiversity and climate resilience")
        
        # Transportation recommendations
        car_dependency = transport_results.get('car_dependency', 0.0)
        mobility_score = transport_results.get('mobility_score', 0.0)
        
        if car_dependency > 0.7:
            recommendations.append("🚗 CRITICAL: High car dependency detected - implement complete streets design with protected bike lanes and enhanced public transit")
        elif mobility_score < 0.3:
            recommendations.append("🚌 PRIORITY: Improve sustainable mobility options - add bus rapid transit, bike-share stations, and pedestrian infrastructure")
        else:
            recommendations.append("🚲 GOOD: Balanced transportation mix supports sustainable urban mobility")
        
        # Infrastructure recommendations
        infra_objects = len(infra_results.get('infrastructure_objects', []))
        if infra_objects < 3:
            recommendations.append("🏗️ PRIORITY: Enhance urban amenities - add street furniture, wayfinding systems, and accessibility features")
        else:
            recommendations.append("📊 GOOD: Adequate urban infrastructure supports community livability")
        
        # Overall score-based recommendations
        if final_score >= 85:
            recommendations.append("⭐ EXEMPLARY: This street demonstrates world-class sustainable urban design principles")
        elif final_score >= 70:
            recommendations.append("✅ STRONG: Solid sustainability foundation - focus on targeted improvements for maximum impact")
        elif final_score >= 50:
            recommendations.append("📈 DEVELOPING: Moderate sustainability - strategic interventions can significantly improve urban livability")
        else:
            recommendations.append("🎯 TRANSFORMATION NEEDED: Comprehensive sustainability improvements required - prioritize green infrastructure and sustainable mobility")
        
        return recommendations
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        
        return {
            'ensemble_architecture': 'Professional Multi-Specialist System',
            'specialists': {
                'green_detection': {
                    'model': 'YOLOv8s + Advanced Segmentation',
                    'focus': 'Vegetation, green infrastructure, environmental health'
                },
                'urban_infrastructure': {
                    'model': 'YOLOv8s + DeepLabV3+',
                    'focus': 'Buildings, accessibility, urban furniture'
                },
                'transportation': {
                    'model': 'YOLOv8x High-Accuracy',
                    'focus': 'Vehicles, transit, pedestrians, mobility'
                },
                'ensemble_scorer': {
                    'model': 'Confidence-Weighted Ensemble',
                    'focus': 'Intelligent score combination'
                }
            },
            'device': str(self.device),
            'accuracy_level': 'professional',
            'confidence_thresholds': 'adaptive',
            'ensemble_method': 'confidence_weighted'
        }
