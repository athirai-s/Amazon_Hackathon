import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import CameraCapture from './components/CameraCapture'
import ResultsScreen from './components/ResultsScreen'
import LoadingScreen from './components/LoadingScreen'

function App() {
  const [currentScreen, setCurrentScreen] = useState('camera') // 'camera', 'loading', 'results'
  const [analysisResult, setAnalysisResult] = useState(null)
  const [capturedImage, setCapturedImage] = useState(null)

  const handleImageCapture = async (imageFile) => {
    setCapturedImage(imageFile)
    setCurrentScreen('loading')
    
    try {
      // Simulate API call to AWS Rekognition
      const result = await analyzeImage(imageFile)
      setAnalysisResult(result)
      setCurrentScreen('results')
    } catch (error) {
      console.error('Analysis failed:', error)
      // For demo purposes, show mock results even if API fails
      const mockResult = generateMockResult()
      setAnalysisResult(mockResult)
      setCurrentScreen('results')
    }
  }

  const handleRetakePhoto = () => {
    setCurrentScreen('camera')
    setAnalysisResult(null)
    setCapturedImage(null)
  }

  const analyzeImage = async (imageFile) => {
    // Mock analysis for demo - in production this would call AWS Rekognition
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve(generateMockResult())
      }, 3000) // 3 second delay to simulate processing
    })
  }

  const generateMockResult = () => {
    // Generate realistic mock data
    const greenCoverage = Math.floor(Math.random() * 30) + 5
    const walkability = Math.floor(Math.random() * 25) + 5
    const transitAccess = Math.floor(Math.random() * 25) + 0
    const carDependency = Math.floor(Math.random() * 15) + 0
    const buildingEfficiency = Math.floor(Math.random() * 20) + 0
    
    const totalScore = Math.min(50 + greenCoverage + walkability + transitAccess - carDependency + buildingEfficiency, 100)
    
    return {
      score: totalScore,
      categories: {
        greenCoverage: {
          score: greenCoverage,
          maxScore: 30,
          detected: ['Tree', 'Plant', 'Grass'],
          count: Math.floor(greenCoverage / 8)
        },
        walkability: {
          score: walkability,
          maxScore: 25,
          detected: ['Sidewalk', 'Person'],
          count: Math.floor(walkability / 5)
        },
        transitAccess: {
          score: transitAccess,
          maxScore: 25,
          detected: transitAccess > 0 ? ['Bus Stop'] : [],
          count: Math.floor(transitAccess / 10)
        },
        carDependency: {
          score: -carDependency,
          maxScore: 0,
          detected: ['Car', 'Parking'],
          count: Math.floor(carDependency / 3)
        },
        buildingEfficiency: {
          score: buildingEfficiency,
          maxScore: 20,
          detected: buildingEfficiency > 10 ? ['Solar Panel'] : [],
          count: Math.floor(buildingEfficiency / 6)
        }
      },
      recommendations: generateRecommendations(totalScore)
    }
  }

  const generateRecommendations = (score) => {
    const recommendations = []
    
    if (score < 60) {
      recommendations.push("Add more trees and green spaces to improve sustainability")
      recommendations.push("Consider adding bike lanes for better transit access")
    } else if (score < 80) {
      recommendations.push("Great foundation! Add solar panels to boost building efficiency")
      recommendations.push("Reduce parking spaces to decrease car dependency")
    } else {
      recommendations.push("Excellent sustainability! This is a model eco-friendly street")
      recommendations.push("Share this example to inspire other neighborhoods")
    }
    
    return recommendations
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50">
      <header className="bg-white shadow-sm border-b border-gray-100">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-eco-green rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-xl">🌱</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">EcoAesthetics</h1>
                <p className="text-sm text-gray-600">Urban Sustainability Scorer</p>
              </div>
            </div>
            {currentScreen === 'results' && (
              <button
                onClick={handleRetakePhoto}
                className="btn-secondary text-sm"
              >
                New Scan
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8">
        <AnimatePresence mode="wait">
          {currentScreen === 'camera' && (
            <motion.div
              key="camera"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <CameraCapture onImageCapture={handleImageCapture} />
            </motion.div>
          )}

          {currentScreen === 'loading' && (
            <motion.div
              key="loading"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.3 }}
            >
              <LoadingScreen image={capturedImage} />
            </motion.div>
          )}

          {currentScreen === 'results' && analysisResult && (
            <motion.div
              key="results"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <ResultsScreen 
                result={analysisResult} 
                image={capturedImage}
                onRetake={handleRetakePhoto}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  )
}

export default App
