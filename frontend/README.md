# Plant Disease Detector - React Frontend

A modern, responsive React frontend for the AI-powered plant disease detection system.

## Features

- 🌱 **Beautiful UI**: Modern, gradient-based design with smooth animations
- 📱 **Responsive**: Works perfectly on desktop, tablet, and mobile devices
- 🖼️ **Drag & Drop**: Easy image upload with drag-and-drop functionality
- 🎯 **Real-time Analysis**: Instant disease detection with confidence scores
- 📊 **Detailed Results**: Shows top 3 predictions with confidence percentages
- ⚡ **Fast Loading**: Optimized for quick image processing

## Technology Stack

- **React 18**: Modern React with hooks
- **Styled Components**: CSS-in-JS for styling
- **React Dropzone**: File upload with drag-and-drop
- **Axios**: HTTP client for API communication
- **React Icons**: Beautiful icons throughout the UI

## Setup Instructions

### Prerequisites
- Node.js (version 14 or higher)
- npm or yarn
- Backend API running on port 8000

### Installation

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start the development server:**
   ```bash
   npm start
   ```

3. **Open your browser:**
   Navigate to `http://localhost:3000`

### Production Build

```bash
npm run build
```

This creates an optimized production build in the `build` folder.

## Usage

1. **Upload Image**: Drag and drop a plant leaf image or click to select
2. **Analyze**: Click "Analyze Plant Disease" button
3. **View Results**: See the disease prediction with confidence score
4. **Try Another**: Click "Analyze Another Plant" to test more images

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)

## API Integration

The frontend communicates with the FastAPI backend running on `http://localhost:8000`. Make sure the backend is running before using the frontend.

### API Endpoints Used:
- `POST /predict` - Analyze single image
- `GET /health` - Check API status
- `GET /classes` - Get supported disease classes

## Project Structure

```
frontend/
├── public/
│   ├── index.html
│   └── manifest.json
├── src/
│   ├── App.js          # Main application component
│   ├── index.js        # React entry point
│   └── index.css       # Global styles
├── package.json
└── README.md
```

## Customization

### Styling
The app uses styled-components for styling. You can customize:
- Colors in the `Container` and `Button` components
- Layout in the `MainContent` component
- Typography in the `Title` and `Subtitle` components

### API Configuration
To change the API endpoint, modify the axios calls in `App.js`:
```javascript
const response = await axios.post('YOUR_API_URL/predict', formData, {
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});
```

## Troubleshooting

### Common Issues

1. **Connection Error**: Make sure the backend API is running on port 8000
2. **Image Upload Issues**: Check that the image format is supported
3. **Build Errors**: Ensure all dependencies are installed with `npm install`

### Development Tips

- Use `npm start` for development with hot reloading
- Check browser console for any JavaScript errors
- Verify API responses in the Network tab
- Test with different image sizes and formats

## Performance

- Optimized bundle size with React production build
- Lazy loading for better performance
- Efficient image handling and preview
- Responsive design for all screen sizes

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge
- Mobile browsers (iOS Safari, Chrome Mobile)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the Plant Disease Detection System.
