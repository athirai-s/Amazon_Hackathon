"""
Image preprocessing and utilities for EcoAesthetics
"""

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
from typing import Tuple, Optional
import io

from utils.logger import setup_logger

logger = setup_logger(__name__)

class ImageProcessor:
    """Handle image preprocessing and enhancement for AI analysis"""
    
    def __init__(self, max_size: Tuple[int, int] = (1024, 1024)):
        """
        Initialize image processor
        
        Args:
            max_size: Maximum image dimensions (width, height)
        """
        self.max_size = max_size
        self.min_size = (224, 224)  # Minimum size for most CNN models
        
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for AI analysis
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        
        logger.info(f"Preprocessing image: {image.size} -> processing...")
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large
        if image.size[0] > self.max_size[0] or image.size[1] > self.max_size[1]:
            image = self._resize_maintain_aspect(image, self.max_size)
        
        # Ensure minimum size
        if image.size[0] < self.min_size[0] or image.size[1] < self.min_size[1]:
            image = self._resize_maintain_aspect(image, self.min_size, upscale=True)
        
        # Enhance image quality
        image = self._enhance_image(image)
        
        logger.info(f"Image preprocessed successfully: final size {image.size}")
        
        return image
    
    def _resize_maintain_aspect(self, image: Image.Image, target_size: Tuple[int, int], upscale: bool = False) -> Image.Image:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: PIL Image object
            target_size: Target (width, height)
            upscale: Whether to upscale smaller images
            
        Returns:
            Resized PIL Image
        """
        
        original_width, original_height = image.size
        target_width, target_height = target_size
        
        # Calculate scaling factor
        if upscale:
            scale = max(target_width / original_width, target_height / original_height)
        else:
            scale = min(target_width / original_width, target_height / original_height)
        
        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Apply image enhancements for better AI analysis
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced PIL Image
        """
        
        # Slight sharpening
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        # Slight contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.05)
        
        # Slight color saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.05)
        
        return image
    
    def to_numpy(self, image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to numpy array
        
        Args:
            image: PIL Image object
            
        Returns:
            Numpy array (H, W, C)
        """
        return np.array(image)
    
    def to_tensor_format(self, image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to tensor format (C, H, W) and normalize
        
        Args:
            image: PIL Image object
            
        Returns:
            Numpy array in tensor format, normalized to [0, 1]
        """
        
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Convert from (H, W, C) to (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
        
        return img_array
    
    def extract_color_features(self, image: Image.Image) -> dict:
        """
        Extract color-based features from image
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary of color features
        """
        
        img_array = np.array(image)
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Calculate color statistics
        features = {
            # RGB statistics
            'mean_rgb': np.mean(img_array, axis=(0, 1)).tolist(),
            'std_rgb': np.std(img_array, axis=(0, 1)).tolist(),
            
            # HSV statistics
            'mean_hsv': np.mean(hsv, axis=(0, 1)).tolist(),
            'std_hsv': np.std(hsv, axis=(0, 1)).tolist(),
            
            # Green vegetation indicators
            'green_ratio': self._calculate_green_ratio(img_array),
            'vegetation_index': self._calculate_vegetation_index(img_array),
            
            # Urban indicators
            'gray_ratio': self._calculate_gray_ratio(img_array),
            'blue_sky_ratio': self._calculate_blue_sky_ratio(img_array, hsv),
        }
        
        return features
    
    def _calculate_green_ratio(self, img_array: np.ndarray) -> float:
        """Calculate ratio of green pixels (vegetation indicator)"""
        
        # Green pixels: G > R and G > B, with sufficient intensity
        green_mask = (
            (img_array[:, :, 1] > img_array[:, :, 0]) &  # G > R
            (img_array[:, :, 1] > img_array[:, :, 2]) &  # G > B
            (img_array[:, :, 1] > 50)  # Sufficient green intensity
        )
        
        return np.sum(green_mask) / img_array.shape[0] / img_array.shape[1]
    
    def _calculate_vegetation_index(self, img_array: np.ndarray) -> float:
        """Calculate normalized difference vegetation index (NDVI) approximation"""
        
        # Simplified NDVI using RGB (not as accurate as NIR, but useful)
        r = img_array[:, :, 0].astype(np.float32)
        g = img_array[:, :, 1].astype(np.float32)
        
        # Avoid division by zero
        denominator = r + g + 1e-8
        ndvi = (g - r) / denominator
        
        # Return mean NDVI for vegetation areas (positive values)
        vegetation_ndvi = ndvi[ndvi > 0]
        return np.mean(vegetation_ndvi) if len(vegetation_ndvi) > 0 else 0.0
    
    def _calculate_gray_ratio(self, img_array: np.ndarray) -> float:
        """Calculate ratio of gray/concrete pixels (urban indicator)"""
        
        # Gray pixels: similar R, G, B values
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        
        # Calculate color differences
        rg_diff = np.abs(r.astype(np.int16) - g.astype(np.int16))
        rb_diff = np.abs(r.astype(np.int16) - b.astype(np.int16))
        gb_diff = np.abs(g.astype(np.int16) - b.astype(np.int16))
        
        # Gray pixels have small color differences
        gray_mask = (rg_diff < 30) & (rb_diff < 30) & (gb_diff < 30)
        
        return np.sum(gray_mask) / img_array.shape[0] / img_array.shape[1]
    
    def _calculate_blue_sky_ratio(self, img_array: np.ndarray, hsv: np.ndarray) -> float:
        """Calculate ratio of blue sky pixels"""
        
        # Blue sky: high blue value, low saturation in upper part of image
        upper_half = img_array[:img_array.shape[0]//2, :, :]
        upper_hsv = hsv[:hsv.shape[0]//2, :, :]
        
        # Blue sky conditions
        blue_mask = (
            (upper_half[:, :, 2] > upper_half[:, :, 0]) &  # B > R
            (upper_half[:, :, 2] > upper_half[:, :, 1]) &  # B > G
            (upper_half[:, :, 2] > 100) &  # Sufficient blue intensity
            (upper_hsv[:, :, 1] < 100)  # Low saturation (not too vivid)
        )
        
        return np.sum(blue_mask) / upper_half.shape[0] / upper_half.shape[1]
