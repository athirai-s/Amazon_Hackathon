import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

const ResultsScreen = ({ result, image, onRetake }) => {
  const [imageUrl, setImageUrl] = useState(null)
  const [animatedScore, setAnimatedScore] = useState(0)

  useEffect(() => {
    if (image) {
      const url = URL.createObjectURL(image)
      setImageUrl(url)
      return () => URL.revokeObjectURL(url)
    }
  }, [image])

  useEffect(() => {
    // Animate score counting up
    const duration = 2000 // 2 seconds
    const steps = 60
    const increment = result.score / steps
    let current = 0

    const timer = setInterval(() => {
      current += increment
      if (current >= result.score) {
        setAnimatedScore(result.score)
        clearInterval(timer)
      } else {
        setAnimatedScore(Math.floor(current))
      }
    }, duration / steps)

    return () => clearInterval(timer)
  }, [result.score])

  const getScoreColor = (score) => {
    if (score >= 80) return 'from-green-400 to-green-600'
    if (score >= 60) return 'from-yellow-400 to-orange-500'
    return 'from-orange-500 to-red-500'
  }

  const getScoreLabel = (score) => {
    if (score >= 80) return 'Excellent'
    if (score >= 60) return 'Good'
    if (score >= 40) return 'Fair'
    return 'Needs Improvement'
  }

  const categoryIcons = {
    greenCoverage: '🌳',
    walkability: '🚶‍♀️',
    transitAccess: '🚌',
    carDependency: '🚗',
    buildingEfficiency: '🏢'
  }

  const categoryNames = {
    greenCoverage: 'Green Coverage',
    walkability: 'Walkability',
    transitAccess: 'Transit Access',
    carDependency: 'Car Dependency',
    buildingEfficiency: 'Building Efficiency'
  }

  const getCategoryColor = (categoryKey, score) => {
    if (categoryKey === 'carDependency') {
      // Negative scores are good for car dependency
      if (score >= -5) return 'bg-green-500'
      if (score >= -10) return 'bg-yellow-500'
      return 'bg-red-500'
    } else {
      if (score >= 15) return 'bg-green-500'
      if (score >= 8) return 'bg-yellow-500'
      return 'bg-red-500'
    }
  }

  const getCategoryWidth = (categoryKey, score, maxScore) => {
    if (categoryKey === 'carDependency') {
      // For car dependency, we show the inverse (less is better)
      return Math.max(0, (15 + score) / 15 * 100) // Convert negative to positive percentage
    }
    return Math.max(0, (score / maxScore) * 100)
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-8"
      >
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          Sustainability Analysis Complete
        </h2>
        <p className="text-gray-600">
          Here's how your street scores on urban sustainability
        </p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Column - Image and Score */}
        <div className="space-y-6">
          {/* Image */}
          {imageUrl && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="card"
            >
              <img
                src={imageUrl}
                alt="Analyzed street"
                className="w-full h-64 object-cover rounded-lg"
              />
            </motion.div>
          )}

          {/* Score Circle */}
          <motion.div
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.4, type: "spring", bounce: 0.4 }}
            className="card text-center"
          >
            <h3 className="text-xl font-semibold text-gray-900 mb-4">
              Sustainability Score
            </h3>
            <div className={`score-circle mx-auto bg-gradient-to-br ${getScoreColor(result.score)} animate-score-reveal`}>
              <div className="text-center">
                <div className="text-4xl font-bold">{animatedScore}</div>
                <div className="text-sm opacity-90">/ 100</div>
              </div>
            </div>
            <div className="mt-4">
              <span className={`inline-block px-4 py-2 rounded-full text-white font-semibold bg-gradient-to-r ${getScoreColor(result.score)}`}>
                {getScoreLabel(result.score)}
              </span>
            </div>
          </motion.div>
        </div>

        {/* Right Column - Category Breakdown */}
        <div className="space-y-6">
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
            className="card"
          >
            <h3 className="text-xl font-semibold text-gray-900 mb-6">
              Category Breakdown
            </h3>
            
            <div className="space-y-4">
              {Object.entries(result.categories).map(([key, category], index) => (
                <motion.div
                  key={key}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.8 + index * 0.1 }}
                  className="space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <span className="text-2xl">{categoryIcons[key]}</span>
                      <span className="font-medium text-gray-900">
                        {categoryNames[key]}
                      </span>
                    </div>
                    <div className="text-right">
                      <span className="font-semibold text-gray-900">
                        {category.score > 0 ? '+' : ''}{category.score}
                      </span>
                      <span className="text-gray-500 text-sm ml-1">
                        / {key === 'carDependency' ? '0' : category.maxScore}
                      </span>
                    </div>
                  </div>
                  
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <motion.div
                      className={`category-bar ${getCategoryColor(key, category.score)}`}
                      initial={{ width: 0 }}
                      animate={{ width: `${getCategoryWidth(key, category.score, category.maxScore)}%` }}
                      transition={{ duration: 1, delay: 1 + index * 0.1 }}
                    />
                  </div>
                  
                  {category.detected.length > 0 && (
                    <div className="text-sm text-gray-600">
                      Detected: {category.detected.join(', ')} ({category.count} items)
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Recommendations */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 1.2 }}
            className="card bg-blue-50 border-blue-200"
          >
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              💡 Improvement Suggestions
            </h3>
            <ul className="space-y-2">
              {result.recommendations.map((recommendation, index) => (
                <motion.li
                  key={index}
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: 1.4 + index * 0.1 }}
                  className="flex items-start space-x-2 text-sm text-gray-700"
                >
                  <span className="text-blue-500 mt-1">•</span>
                  <span>{recommendation}</span>
                </motion.li>
              ))}
            </ul>
          </motion.div>
        </div>
      </div>

      {/* Action Buttons */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 1.6 }}
        className="mt-8 flex flex-col sm:flex-row gap-4 justify-center"
      >
        <button
          onClick={onRetake}
          className="btn-secondary"
        >
          📸 Scan Another Street
        </button>
        <button
          onClick={() => {
            // Mock share functionality
            if (navigator.share) {
              navigator.share({
                title: 'EcoAesthetics Score',
                text: `This street scored ${result.score}/100 on sustainability!`,
                url: window.location.href
              })
            } else {
              // Fallback - copy to clipboard
              navigator.clipboard.writeText(`This street scored ${result.score}/100 on sustainability! Check it out on EcoAesthetics.`)
              alert('Score copied to clipboard!')
            }
          }}
          className="btn-primary"
        >
          📤 Share Score
        </button>
      </motion.div>

      {/* Additional Info */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 1.8 }}
        className="mt-8 card bg-green-50 border-green-200"
      >
        <h4 className="font-semibold text-gray-900 mb-3">🌍 About This Score</h4>
        <p className="text-sm text-gray-600 mb-3">
          Your sustainability score is calculated based on AI analysis of visual elements that contribute to urban livability and environmental impact.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-gray-600">
          <div>
            <strong>Green Coverage:</strong> Trees, plants, and green spaces that improve air quality and reduce heat
          </div>
          <div>
            <strong>Walkability:</strong> Sidewalks, crosswalks, and pedestrian-friendly infrastructure
          </div>
          <div>
            <strong>Transit Access:</strong> Public transportation and bike infrastructure
          </div>
          <div>
            <strong>Building Efficiency:</strong> Solar panels, green roofs, and sustainable architecture
          </div>
        </div>
      </motion.div>
    </div>
  )
}

export default ResultsScreen
