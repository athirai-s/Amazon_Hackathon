# 🌱 EcoAesthetics: Urban Sustainability Scorer

An Instagram-meets-urban-planning web application that analyzes street photos to provide instant sustainability scores using AI-powered computer vision.

## 🎯 Features

### Core Functionality
- **Photo Upload**: Drag-and-drop or click to upload street photos
- **AI Analysis**: Simulated computer vision analysis (ready for AWS Rekognition integration)
- **Sustainability Scoring**: 0-100 score based on multiple categories
- **Beautiful UI**: Modern, responsive design with smooth animations

### Scoring Categories
- 🌳 **Green Coverage** (0-30 points): Trees, plants, gardens, green spaces
- 🚶‍♀️ **Walkability** (0-25 points): Sidewalks, crosswalks, pedestrian areas
- 🚌 **Transit Access** (0-25 points): Bus stops, bike lanes, transit infrastructure
- 🚗 **Car Dependency** (-15 to 0 points): Parking lots, traffic density (negative impact)
- 🏢 **Building Efficiency** (0-20 points): Solar panels, green roofs, sustainable architecture

## 🛠️ Tech Stack

- **Frontend**: React 18 + Vite
- **Styling**: Tailwind CSS with custom animations
- **Animations**: Framer Motion
- **AI Integration**: Ready for AWS Rekognition (currently using mock data)
- **Future**: AWS S3, Lambda, Location Services

## 🚀 Getting Started

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm run dev
   ```

3. **Open Browser**
   Navigate to `http://localhost:3000`

## 📱 User Flow

1. **Landing Page**: Clean interface with sustainability category icons
2. **Photo Upload**: Drag-and-drop or file selection
3. **Analysis**: Animated loading screen with progress steps
4. **Results**: Score reveal with category breakdown and recommendations
5. **Actions**: Share score or scan another street

## 🎨 Design Features

- **Instagram-like UI**: Modern, clean, mobile-first design
- **Smooth Animations**: Score counting, progress bars, transitions
- **Color-coded Results**: Green (excellent) to red (needs improvement)
- **Interactive Elements**: Hover effects, button animations
- **Responsive Layout**: Works on desktop and mobile

## 🔮 Future Enhancements

### Phase 1: AWS Integration
- Real AWS Rekognition for image analysis
- S3 bucket for image storage
- Lambda functions for processing

### Phase 2: Social Features
- User accounts and score history
- Neighborhood leaderboards
- Before/after comparisons
- Community challenges

### Phase 3: Advanced Features
- GPS location tagging
- Interactive maps with score heatmaps
- AR overlay mode
- City planning insights

## 🏗️ Architecture

```
src/
├── components/
│   ├── CameraCapture.jsx    # Photo upload interface
│   ├── LoadingScreen.jsx    # Analysis progress display
│   └── ResultsScreen.jsx    # Score and recommendations
├── App.jsx                  # Main application logic
├── main.jsx                 # React entry point
└── index.css               # Tailwind + custom styles
```

## 🎯 Scoring Algorithm

```javascript
const calculateScore = (detectedElements) => {
  let score = 50; // Base score
  
  // Green Coverage (+8 per tree/plant)
  score += countElements(['Tree', 'Plant', 'Vegetation']) * 8;
  
  // Walkability (+5 per pedestrian feature)
  score += countElements(['Sidewalk', 'Person']) * 5;
  
  // Transit Access (+10 per transit element)
  score += countElements(['Bus', 'Bicycle', 'Bus Stop']) * 10;
  
  // Car Dependency (-3 per vehicle)
  score -= countElements(['Car', 'Parking']) * 3;
  
  // Building Efficiency (+6 per sustainable feature)
  score += countElements(['Solar Panel', 'Green Roof']) * 6;
  
  return Math.min(Math.max(score, 0), 100);
}
```

## 🌍 Impact

EcoAesthetics helps:
- **Urban Planners**: Assess neighborhood sustainability
- **Citizens**: Understand their environment's eco-friendliness
- **Developers**: Make data-driven sustainable design decisions
- **Communities**: Gamify environmental improvements

## 📄 License

MIT License - Feel free to use and modify for your projects!

---

Built with ❤️ for sustainable urban development
