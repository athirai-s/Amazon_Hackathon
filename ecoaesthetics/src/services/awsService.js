// AWS Service for EcoAesthetics
// This file shows how to integrate with AWS Rekognition for real image analysis

import AWS from 'aws-sdk';

// Configure AWS (in production, use environment variables)
const rekognition = new AWS.Rekognition({
  region: 'us-east-1',
  accessKeyId: process.env.REACT_APP_AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.REACT_APP_AWS_SECRET_ACCESS_KEY
});

const s3 = new AWS.S3({
  region: 'us-east-1',
  accessKeyId: process.env.REACT_APP_AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.REACT_APP_AWS_SECRET_ACCESS_KEY
});

/**
 * Upload image to S3 bucket
 * @param {File} imageFile - The image file to upload
 * @returns {Promise<string>} - S3 object key
 */
export const uploadImageToS3 = async (imageFile) => {
  const bucketName = process.env.REACT_APP_S3_BUCKET_NAME || 'ecoaesthetics-images';
  const key = `uploads/${Date.now()}-${imageFile.name}`;
  
  const params = {
    Bucket: bucketName,
    Key: key,
    Body: imageFile,
    ContentType: imageFile.type
  };
  
  try {
    await s3.upload(params).promise();
    return key;
  } catch (error) {
    console.error('Error uploading to S3:', error);
    throw error;
  }
};

/**
 * Analyze image using AWS Rekognition
 * @param {string} s3Key - S3 object key of the uploaded image
 * @returns {Promise<Object>} - Analysis results
 */
export const analyzeImageWithRekognition = async (s3Key) => {
  const bucketName = process.env.REACT_APP_S3_BUCKET_NAME || 'ecoaesthetics-images';
  
  const params = {
    Image: {
      S3Object: {
        Bucket: bucketName,
        Name: s3Key
      }
    },
    MaxLabels: 50,
    MinConfidence: 70
  };
  
  try {
    const response = await rekognition.detectLabels(params).promise();
    return response.Labels;
  } catch (error) {
    console.error('Error analyzing image with Rekognition:', error);
    throw error;
  }
};

/**
 * Calculate sustainability score from detected labels
 * @param {Array} labels - Labels detected by Rekognition
 * @returns {Object} - Sustainability analysis result
 */
export const calculateSustainabilityScore = (labels) => {
  // Define sustainability-related labels and their categories
  const sustainabilityLabels = {
    greenCoverage: ['Tree', 'Plant', 'Vegetation', 'Grass', 'Garden', 'Park', 'Leaf', 'Flora'],
    walkability: ['Sidewalk', 'Walkway', 'Path', 'Person', 'Pedestrian', 'Crosswalk'],
    transitAccess: ['Bus', 'Bicycle', 'Bus Stop', 'Train', 'Subway', 'Transit'],
    carDependency: ['Car', 'Vehicle', 'Truck', 'Parking', 'Traffic', 'Road'],
    buildingEfficiency: ['Solar Panel', 'Green Roof', 'Building', 'Architecture']
  };
  
  // Count detected elements by category
  const detectedElements = {
    greenCoverage: { detected: [], count: 0, score: 0 },
    walkability: { detected: [], count: 0, score: 0 },
    transitAccess: { detected: [], count: 0, score: 0 },
    carDependency: { detected: [], count: 0, score: 0 },
    buildingEfficiency: { detected: [], count: 0, score: 0 }
  };
  
  // Process each detected label
  labels.forEach(label => {
    const labelName = label.Name;
    const confidence = label.Confidence;
    
    // Only consider labels with high confidence
    if (confidence >= 70) {
      Object.keys(sustainabilityLabels).forEach(category => {
        if (sustainabilityLabels[category].includes(labelName)) {
          detectedElements[category].detected.push(labelName);
          detectedElements[category].count++;
        }
      });
    }
  });
  
  // Calculate scores for each category
  let totalScore = 50; // Base score
  
  // Green Coverage (0-30 points)
  const greenScore = Math.min(detectedElements.greenCoverage.count * 8, 30);
  detectedElements.greenCoverage.score = greenScore;
  detectedElements.greenCoverage.maxScore = 30;
  totalScore += greenScore;
  
  // Walkability (0-25 points)
  const walkScore = Math.min(detectedElements.walkability.count * 5, 25);
  detectedElements.walkability.score = walkScore;
  detectedElements.walkability.maxScore = 25;
  totalScore += walkScore;
  
  // Transit Access (0-25 points)
  const transitScore = Math.min(detectedElements.transitAccess.count * 10, 25);
  detectedElements.transitAccess.score = transitScore;
  detectedElements.transitAccess.maxScore = 25;
  totalScore += transitScore;
  
  // Car Dependency (-15 to 0 points - negative impact)
  const carPenalty = Math.max(detectedElements.carDependency.count * -3, -15);
  detectedElements.carDependency.score = carPenalty;
  detectedElements.carDependency.maxScore = 0;
  totalScore += carPenalty;
  
  // Building Efficiency (0-20 points)
  const buildingScore = Math.min(detectedElements.buildingEfficiency.count * 6, 20);
  detectedElements.buildingEfficiency.score = buildingScore;
  detectedElements.buildingEfficiency.maxScore = 20;
  totalScore += buildingScore;
  
  // Ensure score is within 0-100 range
  totalScore = Math.min(Math.max(totalScore, 0), 100);
  
  // Generate recommendations based on score
  const recommendations = generateRecommendations(totalScore, detectedElements);
  
  return {
    score: Math.round(totalScore),
    categories: detectedElements,
    recommendations,
    rawLabels: labels
  };
};

/**
 * Generate improvement recommendations based on analysis
 * @param {number} score - Overall sustainability score
 * @param {Object} categories - Category breakdown
 * @returns {Array} - Array of recommendation strings
 */
const generateRecommendations = (score, categories) => {
  const recommendations = [];
  
  if (categories.greenCoverage.score < 15) {
    recommendations.push("Add more trees and green spaces to improve air quality and reduce urban heat");
  }
  
  if (categories.walkability.score < 10) {
    recommendations.push("Improve pedestrian infrastructure with better sidewalks and crosswalks");
  }
  
  if (categories.transitAccess.score < 10) {
    recommendations.push("Consider adding bike lanes or improving public transit access");
  }
  
  if (categories.carDependency.score < -10) {
    recommendations.push("Reduce car dependency by limiting parking and promoting alternative transport");
  }
  
  if (categories.buildingEfficiency.score < 10) {
    recommendations.push("Encourage sustainable building practices like solar panels and green roofs");
  }
  
  if (score >= 80) {
    recommendations.push("Excellent sustainability! This street is a model for eco-friendly urban design");
  } else if (score >= 60) {
    recommendations.push("Good foundation for sustainability with room for targeted improvements");
  } else {
    recommendations.push("Significant opportunities exist to improve this street's environmental impact");
  }
  
  return recommendations;
};

/**
 * Complete analysis workflow
 * @param {File} imageFile - The image file to analyze
 * @returns {Promise<Object>} - Complete analysis result
 */
export const analyzeStreetSustainability = async (imageFile) => {
  try {
    // Step 1: Upload image to S3
    const s3Key = await uploadImageToS3(imageFile);
    
    // Step 2: Analyze with Rekognition
    const labels = await analyzeImageWithRekognition(s3Key);
    
    // Step 3: Calculate sustainability score
    const result = calculateSustainabilityScore(labels);
    
    return {
      ...result,
      s3Key,
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    console.error('Error in sustainability analysis:', error);
    throw error;
  }
};

// Export individual functions for testing
export {
  rekognition,
  s3,
  generateRecommendations
};
