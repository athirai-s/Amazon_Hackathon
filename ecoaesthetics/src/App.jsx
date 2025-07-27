import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import CameraCapture from './components/CameraCapture'
import ResultsScreen from './components/ResultsScreen'
import LoadingScreen from './components/LoadingScreen'
import { autoAnalyzeStreetSustainability, mockAnalyzeStreetSustainability, testBackendConnection } from './services/backendService'

function App() {
  const [currentScreen, setCurrentScreen] = useState('camera') // 'camera', 'loading', 'results'
  const [analysisResult, setAnalysisResult] = useState(null)
  const [capturedImage, setCapturedImage] = useState(null)
  const [useMockData, setUseMockData] = useState(false) // Toggle for testing

  const handleImageCapture = async (imageFile) => {
    setCapturedImage(imageFile)
    setCurrentScreen('loading')
    
    try {
      // Use auto-detection for best analysis method, or force mock if toggle is on
      const result = useMockData 
        ? await mockAnalyzeStreetSustainability(imageFile)
        : await autoAnalyzeStreetSustainability(imageFile)
      
      setAnalysisResult(result)
      setCurrentScreen('results')
    } catch (error) {
      console.error('Real AWS analysis failed, falling back to mock:', error)
      // Fallback to mock analysis if AWS fails
      try {
        const mockResult = await mockAnalyzeStreetSustainability(imageFile)
        setAnalysisResult(mockResult)
        setCurrentScreen('results')
      } catch (mockError) {
        console.error('Mock analysis also failed:', mockError)
        // Last resort: generate basic mock data
        const basicMockResult = generateBasicMockResult()
        setAnalysisResult(basicMockResult)
        setCurrentScreen('results')
      }
    }
  }

  const handleRetakePhoto = () => {
    setCurrentScreen('camera')
    setAnalysisResult(null)
    setCapturedImage(null)
  }

  const toggleMockData = () => {
    setUseMockData(!useMockData)
  }

  // Basic fallback for when everything else fails
  const generateBasicMockResult = () => {
    const score = Math.floor(Math.random() * 40) + 40 // 40-80 range
    
    return {
      score,
      method: 'fallback',
      features: {
        greenCoverage: { 
          count: 2, 
          detected: ['Tree', 'Grass'],
          confidence: 85,
          labels: [{ name: 'Tree', confidence: 85 }, { name: 'Grass', confidence: 80 }]
        },
        walkability: { 
          count: 1, 
          detected: ['Sidewalk'],
          confidence: 75,
          labels: [{ name: 'Sidewalk', confidence: 75 }]
        },
        transitAccess: { 
          count: 0, 
          detected: [],
          confidence: 0,
          labels: []
        },
        carDependency: { 
          count: 3, 
          detected: ['Car'],
          confidence: 90,
          labels: [{ name: 'Car', confidence: 90 }]
        },
        buildingEfficiency: { 
          count: 1, 
          detected: ['Building'],
          confidence: 70,
          labels: [{ name: 'Building', confidence: 70 }]
        },
        infrastructure: { 
          count: 1, 
          detected: ['Street Light'],
          confidence: 65,
          labels: [{ name: 'Street Light', confidence: 65 }]
        },
        naturalElements: { 
          count: 1, 
          detected: ['Sky'],
          confidence: 95,
          labels: [{ name: 'Sky', confidence: 95 }]
        }
      },
      categoryScores: {
        greenCoverage: 12,
        walkability: 5,
        transitAccess: 0,
        carDependency: -12,
        buildingEfficiency: 4,
        naturalElements: 2
      },
      recommendations: [
        "🌱 This is a basic analysis - connect to AWS for detailed insights",
        "📊 Enable real AI analysis for comprehensive sustainability scoring",
        "🔧 Configure your AWS credentials to unlock full AI-powered analysis"
      ],
      rawLabels: [
        { Name: 'Tree', Confidence: 85 },
        { Name: 'Grass', Confidence: 80 },
        { Name: 'Sidewalk', Confidence: 75 },
        { Name: 'Car', Confidence: 90 },
        { Name: 'Building', Confidence: 70 },
        { Name: 'Sky', Confidence: 95 }
      ],
      isFallbackData: true,
      timestamp: new Date().toISOString()
    }
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
            <div className="flex items-center space-x-3">
              {/* Developer toggle for testing */}
              <div className="flex items-center space-x-2">
                <label className="text-xs text-gray-500">Mock Data:</label>
                <button
                  onClick={toggleMockData}
                  className={`w-10 h-5 rounded-full transition-colors ${
                    useMockData ? 'bg-blue-500' : 'bg-gray-300'
                  }`}
                >
                  <div
                    className={`w-4 h-4 bg-white rounded-full transition-transform ${
                      useMockData ? 'translate-x-5' : 'translate-x-0.5'
                    }`}
                  />
                </button>
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
