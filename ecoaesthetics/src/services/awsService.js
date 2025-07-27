// AWS Service for EcoAesthetics - Modern SDK v3 Implementation
// This file integrates with AWS Rekognition and S3 for real image analysis

import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { RekognitionClient, DetectLabelsCommand } from '@aws-sdk/client-rekognition';
import { SageMakerRuntimeClient, InvokeEndpointCommand } from '@aws-sdk/client-sagemaker-runtime';

// AWS Configuration
const AWS_CONFIG = {
  region: import.meta.env.VITE_AWS_REGION || 'us-east-1',
  credentials: {
    accessKeyId: import.meta.env.VITE_AWS_ACCESS_KEY_ID,
    secretAccessKey: import.meta.env.VITE_AWS_SECRET_ACCESS_KEY,
  },
};

// Initialize AWS clients
const s3Client = new S3Client(AWS_CONFIG);
const rekognitionClient = new RekognitionClient(AWS_CONFIG);
const sagemakerClient = new SageMakerRuntimeClient(AWS_CONFIG);

// Configuration constants
const BUCKET_NAME = import.meta.env.VITE_S3_BUCKET_NAME || 'ecoaesthetics-images-546727414005';
const SAGEMAKER_ENDPOINT = import.meta.env.VITE_SAGEMAKER_ENDPOINT || null;

/**
 * Upload image to S3 bucket
 * @param {File} imageFile - The image file to upload
 * @returns {Promise<string>} - S3 object key
 */
export const uploadImageToS3 = async (imageFile) => {
  const key = `uploads/${Date.now()}-${imageFile.name.replace(/[^a-zA-Z0-9.-]/g, '_')}`;
  
  try {
    // Convert File to ArrayBuffer for browser compatibility
    const arrayBuffer = await imageFile.arrayBuffer();
    const uint8Array = new Uint8Array(arrayBuffer);
    
    const command = new PutObjectCommand({
      Bucket: BUCKET_NAME,
      Key: key,
      Body: uint8Array,
      ContentType: imageFile.type,
      Metadata: {
        'upload-timestamp': new Date().toISOString(),
        'original-name': imageFile.name,
        'file-size': imageFile.size.toString(),
      },
    });
    
    await s3Client.send(command);
    console.log(`Image uploaded successfully to S3: ${key}`);
    return key;
  } catch (error) {
    console.error('Error uploading to S3:', error);
    throw new Error(`Failed to upload image: ${error.message}`);
  }
};

/**
 * Analyze image using AWS Rekognition
 * @param {string} s3Key - S3 object key of the uploaded image
 * @returns {Promise<Array>} - Array of detected labels
 */
export const analyzeImageWithRekognition = async (s3Key) => {
  const command = new DetectLabelsCommand({
    Image: {
      S3Object: {
        Bucket: BUCKET_NAME,
        Name: s3Key,
      },
    },
    MaxLabels: 50,
    MinConfidence: 70,
    Features: ['GENERAL_LABELS', 'IMAGE_PROPERTIES'],
  });
  
  try {
    const response = await rekognitionClient.send(command);
    console.log(`Rekognition analysis completed for ${s3Key}:`, response.Labels?.length || 0, 'labels detected');
    return response.Labels || [];
  } catch (error) {
    console.error('Error analyzing image with Rekognition:', error);
    throw new Error(`Failed to analyze image: ${error.message}`);
  }
};

/**
 * Call custom SageMaker model for sustainability scoring (if available)
 * @param {Object} features - Extracted features from Rekognition
 * @returns {Promise<Object>} - SageMaker model prediction
 */
export const callSageMakerModel = async (features) => {
  if (!SAGEMAKER_ENDPOINT) {
    console.log('SageMaker endpoint not configured, using fallback scoring');
    return null;
  }
  
  const payload = JSON.stringify({
    instances: [features]
  });
  
  const command = new InvokeEndpointCommand({
    EndpointName: SAGEMAKER_ENDPOINT,
    ContentType: 'application/json',
    Body: payload,
  });
  
  try {
    const response = await sagemakerClient.send(command);
    const result = JSON.parse(new TextDecoder().decode(response.Body));
    console.log('SageMaker prediction completed:', result);
    return result;
  } catch (error) {
    console.error('Error calling SageMaker model:', error);
    // Don't throw error, fall back to rule-based scoring
    return null;
  }
};

/**
 * Extract sustainability-relevant features from Rekognition labels
 * @param {Array} labels - Labels detected by Rekognition
 * @returns {Object} - Structured feature data
 */
export const extractSustainabilityFeatures = (labels) => {
  // Define sustainability-related labels and their categories
  const sustainabilityLabels = {
    greenCoverage: ['Tree', 'Plant', 'Vegetation', 'Grass', 'Garden', 'Park', 'Leaf', 'Flora', 'Forest', 'Bush', 'Flower'],
    walkability: ['Sidewalk', 'Walkway', 'Path', 'Person', 'Pedestrian', 'Crosswalk', 'Walking', 'Footpath'],
    transitAccess: ['Bus', 'Bicycle', 'Bus Stop', 'Train', 'Subway', 'Transit', 'Bike', 'Public Transport'],
    carDependency: ['Car', 'Vehicle', 'Truck', 'Parking', 'Traffic', 'Road', 'Automobile', 'Parking Lot'],
    buildingEfficiency: ['Solar Panel', 'Green Roof', 'Building', 'Architecture', 'Sustainable', 'Modern'],
    infrastructure: ['Street Light', 'Bench', 'Sign', 'Traffic Light', 'Pole', 'Urban'],
    naturalElements: ['Sky', 'Cloud', 'Water', 'River', 'Lake', 'Nature'],
  };
  
  // Initialize feature structure
  const features = {};
  Object.keys(sustainabilityLabels).forEach(category => {
    features[category] = {
      detected: [],
      count: 0,
      confidence: 0,
      labels: []
    };
  });
  
  // Process each detected label
  labels.forEach(label => {
    const labelName = label.Name;
    const confidence = label.Confidence;
    
    // Only consider labels with reasonable confidence
    if (confidence >= 60) {
      Object.keys(sustainabilityLabels).forEach(category => {
        if (sustainabilityLabels[category].some(keyword => 
          labelName.toLowerCase().includes(keyword.toLowerCase()) ||
          keyword.toLowerCase().includes(labelName.toLowerCase())
        )) {
          features[category].detected.push(labelName);
          features[category].count++;
          features[category].confidence = Math.max(features[category].confidence, confidence);
          features[category].labels.push({ name: labelName, confidence });
        }
      });
    }
  });
  
  return features;
};

/**
 * Calculate sustainability score using hybrid approach
 * @param {Array} labels - Labels detected by Rekognition
 * @param {Object} sagemakerResult - Optional SageMaker model result
 * @returns {Object} - Complete sustainability analysis
 */
export const calculateSustainabilityScore = async (labels, sagemakerResult = null) => {
  // Extract features from Rekognition
  const features = extractSustainabilityFeatures(labels);
  
  // If SageMaker model is available, use it for scoring
  if (sagemakerResult && sagemakerResult.predictions) {
    const mlScore = sagemakerResult.predictions[0];
    return {
      score: Math.round(mlScore * 100),
      method: 'machine_learning',
      features,
      recommendations: generateRecommendations(mlScore * 100, features),
      rawLabels: labels,
      sagemakerResult
    };
  }
  
  // Fallback to rule-based scoring
  let totalScore = 50; // Base score
  const categoryScores = {};
  
  // Green Coverage (0-30 points)
  const greenScore = Math.min(features.greenCoverage.count * 6, 30);
  categoryScores.greenCoverage = greenScore;
  totalScore += greenScore;
  
  // Walkability (0-25 points)
  const walkScore = Math.min(features.walkability.count * 5, 25);
  categoryScores.walkability = walkScore;
  totalScore += walkScore;
  
  // Transit Access (0-25 points)
  const transitScore = Math.min(features.transitAccess.count * 8, 25);
  categoryScores.transitAccess = transitScore;
  totalScore += transitScore;
  
  // Car Dependency (-20 to 0 points - negative impact)
  const carPenalty = Math.max(features.carDependency.count * -4, -20);
  categoryScores.carDependency = carPenalty;
  totalScore += carPenalty;
  
  // Building Efficiency (0-20 points)
  const buildingScore = Math.min(features.buildingEfficiency.count * 4, 20);
  categoryScores.buildingEfficiency = buildingScore;
  totalScore += buildingScore;
  
  // Natural Elements bonus (0-10 points)
  const naturalScore = Math.min(features.naturalElements.count * 2, 10);
  categoryScores.naturalElements = naturalScore;
  totalScore += naturalScore;
  
  // Ensure score is within 0-100 range
  totalScore = Math.min(Math.max(totalScore, 0), 100);
  
  return {
    score: Math.round(totalScore),
    method: 'rule_based',
    categoryScores,
    features,
    recommendations: generateRecommendations(totalScore, features),
    rawLabels: labels
  };
};

/**
 * Generate improvement recommendations based on analysis
 * @param {number} score - Overall sustainability score
 * @param {Object} features - Feature breakdown
 * @returns {Array} - Array of recommendation strings
 */
const generateRecommendations = (score, features) => {
  const recommendations = [];
  
  // Green coverage recommendations
  if (features.greenCoverage.count < 3) {
    recommendations.push("🌳 Add more trees and green spaces to improve air quality and reduce urban heat island effect");
  }
  
  // Walkability recommendations
  if (features.walkability.count < 2) {
    recommendations.push("🚶‍♀️ Improve pedestrian infrastructure with better sidewalks, crosswalks, and walking paths");
  }
  
  // Transit recommendations
  if (features.transitAccess.count < 1) {
    recommendations.push("🚌 Consider adding bike lanes, bus stops, or other public transit infrastructure");
  }
  
  // Car dependency recommendations
  if (features.carDependency.count > 5) {
    recommendations.push("🚗 Reduce car dependency by limiting parking spaces and promoting alternative transportation");
  }
  
  // Building efficiency recommendations
  if (features.buildingEfficiency.count < 2) {
    recommendations.push("🏢 Encourage sustainable building practices like solar panels, green roofs, and energy-efficient design");
  }
  
  // Overall score recommendations
  if (score >= 85) {
    recommendations.push("✨ Excellent sustainability! This street is a model for eco-friendly urban design");
  } else if (score >= 70) {
    recommendations.push("👍 Good sustainability foundation with opportunities for targeted improvements");
  } else if (score >= 50) {
    recommendations.push("⚡ Moderate sustainability - several key improvements could make a significant impact");
  } else {
    recommendations.push("🎯 Significant opportunities exist to transform this street's environmental impact");
  }
  
  return recommendations;
};

/**
 * Complete analysis workflow - Hybrid Rekognition + SageMaker approach
 * @param {File} imageFile - The image file to analyze
 * @returns {Promise<Object>} - Complete analysis result
 */
export const analyzeStreetSustainability = async (imageFile) => {
  try {
    console.log('Starting sustainability analysis for:', imageFile.name);
    
    // Step 1: Upload image to S3
    console.log('Step 1: Uploading image to S3...');
    const s3Key = await uploadImageToS3(imageFile);
    
    // Step 2: Analyze with Rekognition
    console.log('Step 2: Analyzing with Rekognition...');
    const labels = await analyzeImageWithRekognition(s3Key);
    
    // Step 3: Extract features for potential SageMaker input
    console.log('Step 3: Extracting sustainability features...');
    const features = extractSustainabilityFeatures(labels);
    
    // Step 4: Try SageMaker model (if available)
    console.log('Step 4: Attempting SageMaker analysis...');
    const sagemakerResult = await callSageMakerModel(features);
    
    // Step 5: Calculate final sustainability score
    console.log('Step 5: Calculating sustainability score...');
    const result = await calculateSustainabilityScore(labels, sagemakerResult);
    
    const finalResult = {
      ...result,
      s3Key,
      imageUrl: `https://${BUCKET_NAME}.s3.${AWS_CONFIG.region}.amazonaws.com/${s3Key}`,
      timestamp: new Date().toISOString(),
      processingTime: Date.now(),
    };
    
    console.log('Analysis completed successfully:', finalResult.score);
    return finalResult;
    
  } catch (error) {
    console.error('Error in sustainability analysis:', error);
    throw new Error(`Analysis failed: ${error.message}`);
  }
};

/**
 * Mock analysis for testing without AWS credentials
 * @param {File} imageFile - The image file to analyze
 * @returns {Promise<Object>} - Mock analysis result
 */
export const mockAnalyzeStreetSustainability = async (imageFile) => {
  // Simulate processing time
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  const mockLabels = [
    { Name: 'Tree', Confidence: 95.5 },
    { Name: 'Building', Confidence: 89.2 },
    { Name: 'Car', Confidence: 76.8 },
    { Name: 'Sidewalk', Confidence: 82.1 },
    { Name: 'Person', Confidence: 71.3 },
    { Name: 'Road', Confidence: 94.7 },
  ];
  
  const features = extractSustainabilityFeatures(mockLabels);
  const result = await calculateSustainabilityScore(mockLabels);
  
  return {
    ...result,
    s3Key: 'mock-key-' + Date.now(),
    imageUrl: URL.createObjectURL(imageFile),
    timestamp: new Date().toISOString(),
    processingTime: 2000,
    isMockData: true,
  };
};

// Export configuration and clients for testing
export {
  s3Client,
  rekognitionClient,
  sagemakerClient,
  AWS_CONFIG,
  BUCKET_NAME,
};
