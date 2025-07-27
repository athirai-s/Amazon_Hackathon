# 🚀 EcoAesthetics - Quick Start Guide

## Current Status ✅
- **Backend**: Running on http://localhost:8000
- **Frontend**: Running on http://localhost:3003
- **AI Models**: DeepLab segmentation + color-based analysis

## How to Use Right Now

### 1. Open the App
Visit **http://localhost:3003** in your browser

### 2. Test the System
- **Mock Data Toggle**: Use the switch in the top-right to test without AI
- **Real AI Analysis**: Turn OFF the toggle to use Python CNN models

### 3. Analyze a Street Photo
1. Click "Take Photo" or "Upload Image"
2. Capture/select a street scene image
3. Wait 2-10 seconds for AI analysis
4. View sustainability score (0-100) with detailed breakdown

## What the AI Analyzes

### Sustainability Categories:
- 🌳 **Green Coverage** (30 pts): Trees, plants, vegetation
- 🚶‍♀️ **Walkability** (25 pts): Sidewalks, crosswalks, pedestrians  
- 🚌 **Transit Access** (25 pts): Public transport, bike lanes
- 🚗 **Car Dependency** (-20 pts): Vehicle density (penalty)
- 🏢 **Building Efficiency** (20 pts): Solar panels, green architecture
- 🏗️ **Infrastructure** (10 pts): Street lights, benches, signs

### AI Models Used:
- **DeepLab v3**: Semantic segmentation for pixel-level analysis
- **Color Analysis**: NDVI vegetation index, urban indicators
- **Fallback Detection**: Color-based object detection when needed

## If You Need to Restart

### Backend Server:
```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```

### Frontend App:
```bash
cd ecoaesthetics
npm run dev
```

## Troubleshooting

### Backend Issues:
- Check if running: `curl http://localhost:8000/health`
- View logs in the terminal running `python main.py`
- Models download automatically on first run (~200MB)

### Frontend Issues:
- Check if running: visit http://localhost:3003
- Toggle "Mock Data" ON if backend fails
- Check browser console for errors

### Common Solutions:
- **Port conflicts**: Frontend will auto-find available port
- **Model loading**: First run takes longer (downloads models)
- **Image upload fails**: Try smaller images (<5MB)

## Features Working Now

✅ **Camera capture** from browser  
✅ **File upload** for existing images  
✅ **Real AI analysis** with CNN models  
✅ **Mock data fallback** for testing  
✅ **Sustainability scoring** with detailed breakdown  
✅ **Recommendations** for improvement  
✅ **Responsive design** works on mobile  
✅ **Error handling** with graceful fallbacks  

## Next Steps (Optional)

### Improve AI Models:
```bash
cd backend
source venv/bin/activate
pip install timm  # For better object detection (DETR model)
```

### Deploy to Cloud:
- **Backend**: Heroku, Railway, Google Cloud Run
- **Frontend**: Vercel, Netlify, AWS Amplify

---

**🎉 Your EcoAesthetics app is ready to analyze street sustainability!**

Visit **http://localhost:3003** to start using it right now.
