# EcoAesthetics - Urban Sustainability Scorer

A React web application that analyzes street photos to provide sustainability scores using Amazon Rekognition and custom machine learning models.

## 🌟 Features

- **Camera Integration**: Capture street photos directly in the browser
- **AI-Powered Analysis**: Uses Amazon Rekognition for object detection
- **Sustainability Scoring**: Hybrid approach combining rule-based and ML scoring
- **Real-time Results**: Instant feedback with detailed category breakdowns
- **Recommendations**: Actionable suggestions for improving street sustainability
- **Responsive Design**: Works on desktop and mobile devices

## 🏗️ Architecture

### Hybrid AI Approach
1. **Amazon Rekognition**: Object detection and labeling
2. **Feature Extraction**: Convert labels to sustainability features
3. **Custom SageMaker Model**: Advanced ML scoring (optional)
4. **Rule-based Fallback**: Reliable scoring when ML models unavailable

### Sustainability Categories
- 🌳 **Green Coverage**: Trees, plants, vegetation
- 🚶‍♀️ **Walkability**: Sidewalks, crosswalks, pedestrian areas
- 🚌 **Transit Access**: Public transport, bike lanes
- 🚗 **Car Dependency**: Parking, traffic density (negative impact)
- 🏢 **Building Efficiency**: Solar panels, sustainable architecture

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ and npm
- AWS Account with programmatic access
- Modern web browser with camera support

### Installation

1. **Clone and install dependencies**
```bash
git clone <repository-url>
cd ecoaesthetics
npm install
```

2. **Configure AWS credentials**
```bash
# Create .env file
cp .env.example .env

# Edit .env with your AWS credentials
VITE_AWS_REGION=us-east-1
VITE_AWS_ACCESS_KEY_ID=your_access_key_here
VITE_AWS_SECRET_ACCESS_KEY=your_secret_key_here
VITE_S3_BUCKET_NAME=ecoaesthetics-images-your-account-id
```

3. **Set up AWS services**
```bash
# Create S3 bucket
aws s3 mb s3://ecoaesthetics-images-your-account-id

# Enable Rekognition (no setup required - pay per use)
# SageMaker setup (optional for custom models)
```

4. **Start development server**
```bash
npm run dev
```

## 🔧 AWS Setup Guide

### 1. IAM User Setup
```bash
# Create IAM user with required permissions
aws iam create-user --user-name ecoaesthetics-dev

# Attach necessary policies
aws iam attach-user-policy \
    --user-name ecoaesthetics-dev \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-user-policy \
    --user-name ecoaesthetics-dev \
    --policy-arn arn:aws:iam::aws:policy/AmazonRekognitionFullAccess

aws iam attach-user-policy \
    --user-name ecoaesthetics-dev \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

# Create access keys
aws iam create-access-key --user-name ecoaesthetics-dev
```

### 2. S3 Bucket Configuration
```bash
# Create bucket with proper naming
aws s3 mb s3://ecoaesthetics-images-$(aws sts get-caller-identity --query Account --output text)

# Set up CORS for web uploads (optional)
aws s3api put-bucket-cors --bucket your-bucket-name --cors-configuration file://cors.json
```

### 3. Rekognition Setup
- No additional setup required
- Pay-per-use pricing: ~$1-5 per 1000 images
- Automatically detects 1000+ object types

### 4. SageMaker Setup (Optional)
```bash
# Create execution role for SageMaker
aws iam create-role --role-name SageMakerExecutionRole --assume-role-policy-document file://trust-policy.json

# Train custom sustainability model (advanced)
# Deploy model endpoint for real-time inference
```

## 📱 Usage

### Basic Workflow
1. **Open the app** in your browser
2. **Toggle Mock Data** switch (top right) for testing
3. **Capture or upload** a street photo
4. **Wait for analysis** (2-10 seconds)
5. **Review results** with score and recommendations
6. **Take another photo** to compare different streets

### Mock vs Real Data
- **Mock Data ON**: Uses simulated analysis for testing
- **Mock Data OFF**: Uses real AWS Rekognition + ML models
- **Auto-fallback**: Falls back to mock if AWS fails

### Understanding Results
- **Score**: 0-100 sustainability rating
- **Categories**: Breakdown by sustainability factors
- **Recommendations**: Specific improvement suggestions
- **Method**: Shows if using ML model or rule-based scoring

## 🛠️ Development

### Project Structure
```
ecoaesthetics/
├── src/
│   ├── components/          # React components
│   │   ├── CameraCapture.jsx
│   │   ├── LoadingScreen.jsx
│   │   └── ResultsScreen.jsx
│   ├── services/
│   │   └── awsService.js    # AWS integration
│   ├── App.jsx              # Main application
│   └── main.jsx
├── public/
├── .env                     # Environment variables
└── package.json
```

### Key Files
- **`awsService.js`**: AWS SDK integration, analysis logic
- **`App.jsx`**: Main app state and routing
- **Components**: UI components for each screen

### Environment Variables
```bash
# Required for AWS integration
VITE_AWS_REGION=us-east-1
VITE_AWS_ACCESS_KEY_ID=your_key
VITE_AWS_SECRET_ACCESS_KEY=your_secret
VITE_S3_BUCKET_NAME=your_bucket

# Optional for custom ML models
VITE_SAGEMAKER_ENDPOINT=your_endpoint_name
```

### Testing
```bash
# Run with mock data (no AWS required)
# Toggle "Mock Data" switch in UI

# Test real AWS integration
# Ensure .env is configured correctly
# Toggle "Mock Data" switch OFF

# Check browser console for detailed logs
```

## 🚀 Deployment

### Vercel (Recommended)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Set environment variables in Vercel dashboard
# VITE_AWS_REGION, VITE_AWS_ACCESS_KEY_ID, etc.
```

### AWS Amplify
```bash
# Connect GitHub repository
# Configure build settings
# Add environment variables
# Auto-deploy on push
```

### Docker
```bash
# Build image
docker build -t ecoaesthetics .

# Run container
docker run -p 3000:3000 ecoaesthetics
```

## 🔒 Security Considerations

### Production Setup
- **Use AWS Cognito** instead of hardcoded credentials
- **Set up IAM roles** with minimal permissions
- **Enable HTTPS** for all deployments
- **Rotate access keys** regularly

### Environment Variables
```bash
# Never commit .env to version control
echo ".env" >> .gitignore

# Use different credentials for dev/prod
# Set up CI/CD with secure secret management
```

## 📊 Cost Estimation

### AWS Services Pricing
- **S3 Storage**: ~$0.023/GB/month
- **Rekognition**: ~$1-5 per 1000 images
- **SageMaker Inference**: ~$50-200/month (if used)
- **Data Transfer**: Minimal for typical usage

### Monthly Estimates
- **Light usage** (100 images): ~$1-5
- **Medium usage** (1000 images): ~$10-20
- **Heavy usage** (10k images): ~$50-100

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

### Code Style
- Use ESLint and Prettier
- Follow React best practices
- Add JSDoc comments for functions
- Test both mock and real AWS integration

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Troubleshooting

### Common Issues

**AWS Credentials Error**
```bash
# Check credentials are set
aws configure list

# Test AWS access
aws sts get-caller-identity
```

**S3 Upload Fails**
- Check bucket name and region
- Verify IAM permissions
- Check CORS configuration

**Rekognition Analysis Fails**
- Ensure image is valid format (JPG, PNG)
- Check file size < 5MB
- Verify Rekognition is enabled in region

**Mock Data Not Working**
- Check browser console for errors
- Ensure all dependencies installed
- Try refreshing the page

### Getting Help
- Check browser console for detailed error logs
- Review AWS CloudWatch logs
- Open GitHub issue with error details
- Join our Discord community

## 🔮 Future Enhancements

- **Mobile App**: React Native version
- **Geolocation**: Map integration with location tagging
- **Social Features**: Share and compare scores
- **Advanced ML**: Custom computer vision models
- **Batch Processing**: Analyze multiple images
- **API Integration**: Third-party urban data sources

---

Built with ❤️ for sustainable urban development
