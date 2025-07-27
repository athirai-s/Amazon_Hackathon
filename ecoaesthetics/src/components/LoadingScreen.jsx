import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

const LoadingScreen = ({ image }) => {
  const [currentStep, setCurrentStep] = useState(0)
  const [imageUrl, setImageUrl] = useState(null)

  const steps = [
    { text: "Analyzing image composition...", icon: "🔍" },
    { text: "Detecting green spaces and vegetation...", icon: "🌳" },
    { text: "Identifying walkability features...", icon: "🚶‍♀️" },
    { text: "Scanning for transit infrastructure...", icon: "🚌" },
    { text: "Evaluating building efficiency...", icon: "🏢" },
    { text: "Calculating sustainability score...", icon: "📊" }
  ]

  useEffect(() => {
    if (image) {
      const url = URL.createObjectURL(image)
      setImageUrl(url)
      return () => URL.revokeObjectURL(url)
    }
  }, [image])

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev < steps.length - 1) {
          return prev + 1
        }
        return prev
      })
    }, 500)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="max-w-2xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-8"
      >
        <h2 className="text-3xl font-bold text-gray-900 mb-4">
          Analyzing Your Street
        </h2>
        <p className="text-gray-600">
          Our AI is examining your photo for sustainability features...
        </p>
      </motion.div>

      {/* Image Preview */}
      {imageUrl && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="card mb-8"
        >
          <img
            src={imageUrl}
            alt="Uploaded street"
            className="w-full h-64 object-cover rounded-lg"
          />
        </motion.div>
      )}

      {/* Progress Steps */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.4 }}
        className="card"
      >
        <div className="space-y-4">
          {steps.map((step, index) => (
            <motion.div
              key={index}
              className={`flex items-center space-x-4 p-3 rounded-lg transition-all duration-300 ${
                index <= currentStep 
                  ? 'bg-green-50 border border-green-200' 
                  : 'bg-gray-50 border border-gray-200'
              }`}
              initial={{ opacity: 0, x: -20 }}
              animate={{ 
                opacity: index <= currentStep ? 1 : 0.5,
                x: 0
              }}
              transition={{ 
                duration: 0.3,
                delay: index * 0.1
              }}
            >
              <div className={`text-2xl ${index <= currentStep ? 'animate-pulse' : ''}`}>
                {step.icon}
              </div>
              <div className="flex-1">
                <p className={`font-medium ${
                  index <= currentStep ? 'text-green-800' : 'text-gray-600'
                }`}>
                  {step.text}
                </p>
              </div>
              <div className="flex items-center">
                {index < currentStep && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ duration: 0.2 }}
                    className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center"
                  >
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </motion.div>
                )}
                {index === currentStep && (
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    className="w-6 h-6 border-2 border-green-500 border-t-transparent rounded-full"
                  />
                )}
                {index > currentStep && (
                  <div className="w-6 h-6 border-2 border-gray-300 rounded-full" />
                )}
              </div>
            </motion.div>
          ))}
        </div>

        {/* Progress Bar */}
        <div className="mt-6">
          <div className="flex justify-between text-sm text-gray-600 mb-2">
            <span>Progress</span>
            <span>{Math.round(((currentStep + 1) / steps.length) * 100)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <motion.div
              className="bg-eco-green h-2 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </div>
      </motion.div>

      {/* Fun Facts */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.8 }}
        className="mt-8 card bg-blue-50 border-blue-200"
      >
        <h4 className="font-semibold text-gray-900 mb-3">💡 Did You Know?</h4>
        <p className="text-sm text-gray-600">
          Streets with more trees can reduce urban temperatures by up to 9°F and improve air quality by filtering pollutants. 
          Walkable neighborhoods also reduce car dependency and promote healthier lifestyles!
        </p>
      </motion.div>
    </div>
  )
}

export default LoadingScreen
