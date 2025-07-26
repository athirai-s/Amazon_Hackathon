import React, { useRef, useState } from 'react'
import { motion } from 'framer-motion'

const CameraCapture = ({ onImageCapture }) => {
  const fileInputRef = useRef(null)
  const [dragActive, setDragActive] = useState(false)

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      onImageCapture(file)
    }
  }

  const handleFileInputChange = (e) => {
    const file = e.target.files[0]
    handleFileSelect(file)
  }

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0])
    }
  }

  const openFileDialog = () => {
    fileInputRef.current?.click()
  }

  return (
    <div className="max-w-2xl mx-auto">
      {/* Hero Section */}
      <motion.div 
        className="text-center mb-8"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="text-4xl font-bold text-gray-900 mb-4">
          Scan Your Street
        </h2>
        <p className="text-lg text-gray-600 mb-6">
          Take a photo of any city street and get an instant sustainability score
        </p>
        
        {/* Feature highlights */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <motion.div 
            className="text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.5 }}
          >
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-2">
              <span className="text-2xl">🌳</span>
            </div>
            <p className="text-sm text-gray-600">Green Coverage</p>
          </motion.div>
          
          <motion.div 
            className="text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
          >
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-2">
              <span className="text-2xl">🚶‍♀️</span>
            </div>
            <p className="text-sm text-gray-600">Walkability</p>
          </motion.div>
          
          <motion.div 
            className="text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.5 }}
          >
            <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-2">
              <span className="text-2xl">🚌</span>
            </div>
            <p className="text-sm text-gray-600">Transit Access</p>
          </motion.div>
          
          <motion.div 
            className="text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.5 }}
          >
            <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center mx-auto mb-2">
              <span className="text-2xl">🏢</span>
            </div>
            <p className="text-sm text-gray-600">Building Efficiency</p>
          </motion.div>
        </div>
      </motion.div>

      {/* Upload Area */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.3, duration: 0.5 }}
      >
        <div
          className={`card relative border-2 border-dashed transition-all duration-300 ${
            dragActive 
              ? 'border-eco-green bg-green-50 scale-105' 
              : 'border-gray-300 hover:border-eco-green hover:bg-green-50'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <div className="text-center py-12">
            <div className="mb-6">
              <div className="w-20 h-20 bg-eco-green rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                Upload Street Photo
              </h3>
              <p className="text-gray-600 mb-6">
                Drag and drop an image here, or click to select
              </p>
            </div>
            
            <button
              onClick={openFileDialog}
              className="btn-primary mb-4"
            >
              Choose Photo
            </button>
            
            <p className="text-sm text-gray-500">
              Supports JPG, PNG, WebP • Max 10MB
            </p>
          </div>
          
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileInputChange}
            className="hidden"
          />
        </div>
      </motion.div>

      {/* Tips Section */}
      <motion.div
        className="mt-8 card bg-blue-50 border-blue-200"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5, duration: 0.5 }}
      >
        <h4 className="font-semibold text-gray-900 mb-3">📸 Tips for Best Results</h4>
        <ul className="text-sm text-gray-600 space-y-2">
          <li className="flex items-start">
            <span className="text-eco-green mr-2">•</span>
            Capture the full street view including sidewalks and buildings
          </li>
          <li className="flex items-start">
            <span className="text-eco-green mr-2">•</span>
            Ensure good lighting for better AI detection
          </li>
          <li className="flex items-start">
            <span className="text-eco-green mr-2">•</span>
            Include visible elements like trees, transit stops, and pedestrian areas
          </li>
          <li className="flex items-start">
            <span className="text-eco-green mr-2">•</span>
            Avoid blurry or heavily filtered images
          </li>
        </ul>
      </motion.div>
    </div>
  )
}

export default CameraCapture
