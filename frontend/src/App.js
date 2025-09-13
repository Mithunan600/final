import React, { useState, useCallback } from 'react';
import styled from 'styled-components';
import { useDropzone } from 'react-dropzone';
import { FaUpload, FaLeaf, FaSpinner, FaCheckCircle, FaExclamationTriangle } from 'react-icons/fa';
import axios from 'axios';

const Container = styled.div`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 40px;
  color: white;
`;

const Title = styled.h1`
  font-size: 3rem;
  margin-bottom: 10px;
  text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 15px;
`;

const Subtitle = styled.p`
  font-size: 1.2rem;
  opacity: 0.9;
  max-width: 600px;
  margin: 0 auto;
`;

const MainContent = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 30px;
  width: 100%;
  max-width: 800px;
`;

const UploadArea = styled.div`
  width: 100%;
  max-width: 600px;
  height: 300px;
  border: 3px dashed ${props => props.isDragActive ? '#4CAF50' : '#ffffff'};
  border-radius: 15px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: ${props => props.isDragActive ? 'rgba(76, 175, 80, 0.1)' : 'rgba(255, 255, 255, 0.1)'};
  cursor: pointer;
  transition: all 0.3s ease;
  backdrop-filter: blur(10px);
  
  &:hover {
    border-color: #4CAF50;
    background: rgba(76, 175, 80, 0.1);
    transform: translateY(-2px);
  }
`;

const UploadIcon = styled(FaUpload)`
  font-size: 3rem;
  color: white;
  margin-bottom: 20px;
  opacity: 0.8;
`;

const UploadText = styled.p`
  color: white;
  font-size: 1.2rem;
  text-align: center;
  margin: 0;
`;

const FileInput = styled.input`
  display: none;
`;

const Button = styled.button`
  background: linear-gradient(45deg, #4CAF50, #45a049);
  color: white;
  border: none;
  padding: 15px 30px;
  font-size: 1.1rem;
  border-radius: 25px;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 15px rgba(0,0,0,0.2);
  display: flex;
  align-items: center;
  gap: 10px;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

const ImagePreview = styled.div`
  width: 100%;
  max-width: 400px;
  border-radius: 15px;
  overflow: hidden;
  box-shadow: 0 10px 30px rgba(0,0,0,0.3);
  background: white;
`;

const PreviewImage = styled.img`
  width: 100%;
  height: auto;
  display: block;
`;

const ResultCard = styled.div`
  width: 100%;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 15px;
  padding: 30px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  backdrop-filter: blur(10px);
`;

const ResultTitle = styled.h2`
  color: #333;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
`;

const PredictionResult = styled.div`
  background: ${props => props.isHealthy ? '#e8f5e8' : '#fff3cd'};
  border: 2px solid ${props => props.isHealthy ? '#4CAF50' : '#ffc107'};
  border-radius: 10px;
  padding: 20px;
  margin-bottom: 20px;
`;

const DiseaseName = styled.h3`
  color: #333;
  margin-bottom: 10px;
  font-size: 1.5rem;
`;

const Confidence = styled.div`
  font-size: 1.1rem;
  color: #666;
  margin-bottom: 10px;
`;

const ConfidenceBar = styled.div`
  width: 100%;
  height: 20px;
  background: #e0e0e0;
  border-radius: 10px;
  overflow: hidden;
  margin-bottom: 15px;
`;

const ConfidenceFill = styled.div`
  height: 100%;
  background: linear-gradient(45deg, #4CAF50, #45a049);
  width: ${props => props.confidence}%;
  transition: width 0.5s ease;
`;

const TopPredictions = styled.div`
  margin-top: 20px;
`;

const PredictionItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px;
  margin: 5px 0;
  background: #f8f9fa;
  border-radius: 8px;
  border-left: 4px solid #4CAF50;
`;

const LoadingSpinner = styled(FaSpinner)`
  animation: spin 1s linear infinite;
  font-size: 2rem;
  color: white;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const ErrorMessage = styled.div`
  background: #f8d7da;
  color: #721c24;
  border: 1px solid #f5c6cb;
  border-radius: 10px;
  padding: 20px;
  margin: 20px 0;
  display: flex;
  align-items: center;
  gap: 10px;
`;

const StatusCard = styled.div`
  background: rgba(255, 255, 255, 0.9);
  border-radius: 10px;
  padding: 20px;
  text-align: center;
  box-shadow: 0 5px 15px rgba(0,0,0,0.1);
`;

const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setPrediction(null);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif']
    },
    multiple: false
  });

  const predictDisease = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await axios.post('/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setPrediction(response.data.prediction);
      } else {
        setError('Prediction failed. Please try again.');
      }
    } catch (err) {
      console.error('Error:', err);
      setError(err.response?.data?.detail || 'Failed to connect to the server. Please make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const resetApp = () => {
    setSelectedFile(null);
    setPrediction(null);
    setError(null);
    setLoading(false);
  };

  return (
    <Container>
      <Header>
        <Title>
          <FaLeaf />
          Plant Disease Detector
        </Title>
        <Subtitle>
          Upload an image of a plant leaf to detect diseases using AI-powered deep learning.
          Our model has 98.2% accuracy across 15 different plant diseases.
        </Subtitle>
      </Header>

      <MainContent>
        {!selectedFile ? (
          <UploadArea {...getRootProps()} isDragActive={isDragActive}>
            <FileInput {...getInputProps()} />
            <UploadIcon />
            <UploadText>
              {isDragActive
                ? 'Drop the image here...'
                : 'Drag & drop a plant image here, or click to select'}
            </UploadText>
          </UploadArea>
        ) : (
          <>
            <ImagePreview>
              <PreviewImage
                src={URL.createObjectURL(selectedFile)}
                alt="Selected plant image"
              />
            </ImagePreview>

            {!prediction && !loading && !error && (
              <Button onClick={predictDisease}>
                <FaLeaf />
                Analyze Plant Disease
              </Button>
            )}

            {loading && (
              <StatusCard>
                <LoadingSpinner />
                <p style={{ marginTop: '10px', color: '#666' }}>
                  Analyzing plant disease... This may take a few seconds.
                </p>
              </StatusCard>
            )}

            {error && (
              <ErrorMessage>
                <FaExclamationTriangle />
                {error}
              </ErrorMessage>
            )}

            {prediction && (
              <ResultCard>
                <ResultTitle>
                  <FaCheckCircle style={{ color: '#4CAF50' }} />
                  Analysis Results
                </ResultTitle>

                <PredictionResult isHealthy={prediction.is_healthy}>
                  <DiseaseName>
                    {prediction.plant_type} - {prediction.disease}
                  </DiseaseName>
                  <Confidence>
                    Confidence: {(prediction.confidence * 100).toFixed(1)}%
                  </Confidence>
                  <ConfidenceBar>
                    <ConfidenceFill confidence={prediction.confidence * 100} />
                  </ConfidenceBar>
                  <p style={{ color: prediction.is_healthy ? '#2e7d32' : '#f57c00', fontWeight: 'bold' }}>
                    {prediction.is_healthy ? '✅ Healthy Plant' : '⚠️ Disease Detected'}
                  </p>
                </PredictionResult>

                <TopPredictions>
                  <h4>Top 3 Predictions:</h4>
                  {prediction.top_3_predictions.map((pred, index) => (
                    <PredictionItem key={index}>
                      <span>{pred.class}</span>
                      <span>{(pred.confidence * 100).toFixed(1)}%</span>
                    </PredictionItem>
                  ))}
                </TopPredictions>

                <Button onClick={resetApp} style={{ marginTop: '20px' }}>
                  Analyze Another Plant
                </Button>
              </ResultCard>
            )}
          </>
        )}
      </MainContent>
    </Container>
  );
};

export default App;
