// Backend Service for EcoAesthetics
// Connects React frontend to Python FastAPI backend

const API_BASE_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

/**
 * Analyze street image using Python backend with CNN models
 * @param {File} imageFile - The image file to analyze
 * @returns {Promise<Object>} - Analysis result from backend
 */
export const analyzeStreetSustainability = async (imageFile) => {
  try {
    console.log('Sending image to Python backend for analysis...');
    
    // Create FormData for file upload
    const formData = new FormData();
    formData.append('file', imageFile);
    
    // Send request to backend
    const response = await fetch(`${API_BASE_URL}/analyze-sustainability`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }
    
    const result = await response.json();
    console.log('Backend analysis completed:', result);
    
    return result;
    
  } catch (error) {
    console.error('Backend analysis failed:', error);
    throw new Error(`Backend analysis failed: ${error.message}`);
  }
};

/**
 * Analyze multiple images in batch
 * @param {File[]} imageFiles - Array of image files
 * @returns {Promise<Object>} - Batch analysis results
 */
export const analyzeBatchSustainability = async (imageFiles) => {
  try {
    console.log(`Sending ${imageFiles.length} images for batch analysis...`);
    
    const formData = new FormData();
    imageFiles.forEach((file, index) => {
      formData.append('files', file);
    });
    
    const response = await fetch(`${API_BASE_URL}/analyze-batch`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }
    
    const result = await response.json();
    console.log('Batch analysis completed:', result);
    
    return result;
    
  } catch (error) {
    console.error('Batch analysis failed:', error);
    throw new Error(`Batch analysis failed: ${error.message}`);
  }
};

/**
 * Check backend health and model status
 * @returns {Promise<Object>} - Health check result
 */
export const checkBackendHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }
    
    return await response.json();
    
  } catch (error) {
    console.error('Backend health check failed:', error);
    throw error;
  }
};

/**
 * Get information about loaded AI models
 * @returns {Promise<Object>} - Model information
 */
export const getModelInfo = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/models/info`);
    
    if (!response.ok) {
      throw new Error(`Model info request failed: ${response.status}`);
    }
    
    return await response.json();
    
  } catch (error) {
    console.error('Failed to get model info:', error);
    throw error;
  }
};

/**
 * Mock analysis for testing when backend is unavailable
 * @param {File} imageFile - The image file to analyze
 * @returns {Promise<Object>} - Mock analysis result
 */
export const mockAnalyzeStreetSustainability = async (imageFile) => {
  // Simulate processing time
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  const mockResult = {
    score: Math.floor(Math.random() * 40) + 40, // 40-80 range
    method: 'mock_analysis',
    features: {
      greenCoverage: {
        detected: ['tree', 'grass', 'vegetation'],
        count: Math.floor(Math.random() * 5) + 1,
        confidence: 0.85,
        labels: [
          { name: 'tree', confidence: 0.92 },
          { name: 'grass', confidence: 0.78 }
        ]
      },
      walkability: {
        detected: ['person', 'sidewalk'],
        count: Math.floor(Math.random() * 3) + 1,
        confidence: 0.75,
        labels: [
          { name: 'person', confidence: 0.88 },
          { name: 'sidewalk', confidence: 0.62 }
        ]
      },
      transitAccess: {
        detected: Math.random() > 0.5 ? ['bicycle'] : [],
        count: Math.random() > 0.5 ? 1 : 0,
        confidence: 0.65,
        labels: Math.random() > 0.5 ? [{ name: 'bicycle', confidence: 0.65 }] : []
      },
      carDependency: {
        detected: ['car'],
        count: Math.floor(Math.random() * 4) + 1,
        confidence: 0.90,
        labels: [
          { name: 'car', confidence: 0.95 }
        ]
      },
      buildingEfficiency: {
        detected: ['building'],
        count: Math.floor(Math.random() * 2) + 1,
        confidence: 0.70,
        labels: [
          { name: 'building', confidence: 0.70 }
        ]
      },
      infrastructure: {
        detected: ['traffic light', 'bench'],
        count: Math.floor(Math.random() * 3) + 1,
        confidence: 0.60,
        labels: [
          { name: 'traffic light', confidence: 0.65 },
          { name: 'bench', confidence: 0.55 }
        ]
      }
    },
    categoryScores: {
      greenCoverage: Math.floor(Math.random() * 20) + 10,
      walkability: Math.floor(Math.random() * 15) + 5,
      transitAccess: Math.floor(Math.random() * 15),
      carDependency: -(Math.floor(Math.random() * 15) + 5),
      buildingEfficiency: Math.floor(Math.random() * 10) + 5,
      infrastructure: Math.floor(Math.random() * 8) + 2
    },
    recommendations: [
      "🌳 Add more trees and green spaces to improve air quality",
      "🚶‍♀️ Improve pedestrian infrastructure with better sidewalks",
      "🚌 Consider adding bike lanes or public transit access",
      "📊 This is mock data - connect to Python backend for real AI analysis"
    ],
    objects: [
      { label: 'tree', confidence: 0.92, bbox: [100, 50, 200, 150] },
      { label: 'car', confidence: 0.95, bbox: [300, 200, 450, 300] },
      { label: 'person', confidence: 0.88, bbox: [150, 180, 180, 250] }
    ],
    color_features: {
      mean_rgb: [120, 140, 110],
      green_ratio: 0.25,
      vegetation_index: 0.15,
      gray_ratio: 0.35,
      blue_sky_ratio: 0.20
    },
    processing_time: 2.0,
    timestamp: new Date().toISOString(),
    model_info: {
      object_detection: { model: 'Mock Model', loaded: false },
      semantic_segmentation: { model: 'Mock Model', loaded: false },
      device: 'cpu',
      categories: ['greenCoverage', 'walkability', 'transitAccess', 'carDependency', 'buildingEfficiency', 'infrastructure']
    },
    isMockData: true
  };
  
  return mockResult;
};

/**
 * Test backend connectivity
 * @returns {Promise<boolean>} - True if backend is reachable
 */
export const testBackendConnection = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/`, {
      method: 'GET',
      timeout: 5000
    });
    
    return response.ok;
    
  } catch (error) {
    console.warn('Backend connection test failed:', error.message);
    return false;
  }
};

/**
 * Auto-detect best analysis method (backend vs mock)
 * @param {File} imageFile - The image file to analyze
 * @returns {Promise<Object>} - Analysis result
 */
export const autoAnalyzeStreetSustainability = async (imageFile) => {
  try {
    // First, test if backend is available
    const backendAvailable = await testBackendConnection();
    
    if (backendAvailable) {
      console.log('Backend available, using real AI analysis');
      return await analyzeStreetSustainability(imageFile);
    } else {
      console.log('Backend unavailable, using mock analysis');
      return await mockAnalyzeStreetSustainability(imageFile);
    }
    
  } catch (error) {
    console.warn('Real analysis failed, falling back to mock:', error.message);
    return await mockAnalyzeStreetSustainability(imageFile);
  }
};

// Export configuration
export const BACKEND_CONFIG = {
  BASE_URL: API_BASE_URL,
  ENDPOINTS: {
    ANALYZE: '/analyze-sustainability',
    BATCH: '/analyze-batch',
    HEALTH: '/health',
    MODEL_INFO: '/models/info'
  },
  TIMEOUT: 30000, // 30 seconds
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
  SUPPORTED_FORMATS: ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff']
};

export default {
  analyzeStreetSustainability,
  analyzeBatchSustainability,
  checkBackendHealth,
  getModelInfo,
  mockAnalyzeStreetSustainability,
  testBackendConnection,
  autoAnalyzeStreetSustainability,
  BACKEND_CONFIG
};
