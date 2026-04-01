import { useState, useRef, useCallback, useEffect } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

interface Prediction {
  crop: string
  disease: string
  class_name: string
  confidence: number
  is_healthy: boolean
}

interface TopPrediction {
  crop: string
  disease: string
  confidence: number
}

interface PredictionResult {
  success: boolean
  prediction: Prediction
  top_predictions: TopPrediction[]
}

interface CropInfo {
  diseases: string[]
  description: string
  icon: string
}

const DEFAULT_CROPS: Record<string, CropInfo> = {
  Sugarcane: { diseases: ['Red Rot', 'Smut', 'Rust'], description: 'Common sugarcane leaf diseases', icon: '🌾' },
  Pulses: { diseases: ['Anthracnose', 'Powdery Mildew', 'Rust'], description: 'Common pulse crop diseases', icon: '🫘' },
  Maize: { diseases: ['Northern Leaf Blight', 'Common Rust', 'Gray Leaf Spot'], description: 'Common maize/corn diseases', icon: '🌽' },
  Wheat: { diseases: ['Leaf Rust', 'Septoria', 'Yellow Rust'], description: 'Common wheat diseases', icon: '🌾' },
  Paddy: { diseases: ['Blast', 'Brown Spot', 'Leaf Scald'], description: 'Common paddy/rice diseases', icon: '🍚' },
  Mustard: { diseases: ['White Rust', 'Alternaria Blight', 'Downy Mildew'], description: 'Common mustard diseases', icon: '🌿' },
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [crops, setCrops] = useState<Record<string, CropInfo>>(DEFAULT_CROPS)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetch(`${API_URL}/crops`)
      .then(res => res.json())
      .then(data => setCrops(data))
      .catch(() => setCrops(DEFAULT_CROPS))
  }, [])

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload a valid image file (JPG, PNG, WebP)')
      return
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('File too large. Max size is 10MB.')
      return
    }
    setSelectedFile(file)
    setResult(null)
    setError(null)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target?.result as string)
    reader.readAsDataURL(file)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }, [handleFile])

  const analyzeImage = async () => {
    if (!selectedFile) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      const response = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData })

      if (!response.ok) {
        const errData = await response.json().catch(() => null)
        throw new Error(errData?.detail || `Server error (${response.status})`)
      }

      const data: PredictionResult = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze image. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  const clearImage = () => {
    setSelectedFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  return (
    <div className="app">
      {/* Navbar */}
      <nav className="navbar">
        <div className="container">
          <div className="nav-brand">
            <span className="nav-logo">🌱</span>
            <span className="nav-title">CropGuard AI</span>
          </div>
          <span className="nav-badge">v1.0 • ML Powered</span>
        </div>
      </nav>

      {/* Hero */}
      <section className="hero">
        <div className="container">
          <div className="hero-tag">🔬 AI-Powered Disease Detection</div>
          <h1>
            Identify <span className="gradient-text">Crop Diseases</span>
            <br />Instantly with AI
          </h1>
          <p className="hero-subtitle">
            Upload a photo of a crop leaf and our deep learning model will detect diseases
            across sugarcane, pulses, maize, wheat, paddy, and mustard.
          </p>
        </div>
      </section>

      {/* Upload Section */}
      <section className="upload-section">
        <div className="container">
          <div className="upload-card">
            {!preview && !loading && (
              <div
                className={`dropzone ${isDragOver ? 'drag-over' : ''}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current?.click()}
                id="dropzone"
              >
                <span className="dropzone-icon">📸</span>
                <p className="dropzone-title">Drop your crop leaf image here</p>
                <p className="dropzone-subtitle">or click to browse files</p>
                <button className="dropzone-btn" type="button">
                  <span>📁</span> Choose Image
                </button>
                <p className="dropzone-formats">Supports JPG, PNG, WebP • Max 10MB</p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  style={{ display: 'none' }}
                  id="file-input"
                />
              </div>
            )}

            {loading && (
              <div className="loading-overlay">
                <div className="loading-spinner" />
                <p className="loading-text">Analyzing your image...</p>
                <p className="loading-subtext">Running inference through the neural network</p>
              </div>
            )}

            {preview && !loading && (
              <>
                <div className="preview-section">
                  {/* Image preview */}
                  <div className="preview-image-card">
                    <button className="preview-remove" onClick={clearImage} title="Remove image">✕</button>
                    <img src={preview} alt="Uploaded crop leaf" />
                    <p className="preview-filename">{selectedFile?.name}</p>
                  </div>

                  {/* Results */}
                  {result ? (
                    <div className="result-card">
                      <div className="result-header">
                        <span className="result-icon">{result.prediction.is_healthy ? '✅' : '⚠️'}</span>
                        <div>
                          <p className="result-title">Detected Crop</p>
                          <p className="result-crop">{result.prediction.crop}</p>
                        </div>
                      </div>

                      <div className={`result-disease ${result.prediction.is_healthy ? 'healthy' : ''}`}>
                        <p className="result-disease-label">
                          {result.prediction.is_healthy ? 'Status' : 'Disease Detected'}
                        </p>
                        <p className="result-disease-name">
                          {result.prediction.is_healthy ? '🌿 Healthy' : `🦠 ${result.prediction.disease}`}
                        </p>
                      </div>

                      <div className="confidence-section">
                        <div className="confidence-header">
                          <span className="confidence-label">Confidence</span>
                          <span className="confidence-value">{result.prediction.confidence}%</span>
                        </div>
                        <div className="confidence-bar">
                          <div
                            className="confidence-fill"
                            style={{ width: `${result.prediction.confidence}%` }}
                          />
                        </div>
                      </div>

                      {result.top_predictions && result.top_predictions.length > 1 && (
                        <div className="top-predictions">
                          <p className="top-predictions-title">Top Predictions</p>
                          {result.top_predictions.map((pred, i) => (
                            <div className="prediction-item" key={i}>
                              <span className="prediction-name">
                                {pred.crop} — {pred.disease}
                              </span>
                              <span className="prediction-conf">
                                {(pred.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="result-card" style={{ alignItems: 'center', justifyContent: 'center' }}>
                      <span style={{ fontSize: 48, opacity: 0.3 }}>🔍</span>
                      <p style={{ color: 'var(--text-muted)', textAlign: 'center' }}>
                        Click "Analyze Image" to detect diseases in this crop leaf
                      </p>
                    </div>
                  )}
                </div>

                {!result && (
                  <button className="analyze-btn" onClick={analyzeImage} disabled={loading} id="analyze-btn">
                    <span>🧬</span> Analyze Image
                  </button>
                )}

                {result && (
                  <button className="analyze-btn" onClick={clearImage} style={{ background: 'rgba(255,255,255,0.06)', marginTop: 20 }}>
                    <span>🔄</span> Upload Another Image
                  </button>
                )}
              </>
            )}

            {error && (
              <div className="error-card">
                <span className="error-icon">⚠️</span>
                <p className="error-text">{error}</p>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Supported Crops */}
      <section className="crops-section">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Supported Crops & Diseases</h2>
            <p className="section-subtitle">Our model can detect the following diseases across 6 crop types</p>
          </div>

          <div className="crops-grid">
            {Object.entries(crops).map(([name, info]) => (
              <div className="crop-card" key={name}>
                <div className="crop-card-header">
                  <span className="crop-emoji">{info.icon}</span>
                  <h3 className="crop-name">{name}</h3>
                </div>
                <p className="crop-desc">{info.description}</p>
                <div className="crop-diseases">
                  {info.diseases.map((disease) => (
                    <span className="disease-tag" key={disease}>{disease}</span>
                  ))}
                  <span className="disease-tag" style={{
                    background: 'rgba(16, 185, 129, 0.08)',
                    color: '#6ee7b7',
                    borderColor: 'rgba(16, 185, 129, 0.15)'
                  }}>Healthy ✓</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <p>CropGuard AI • Built with MobileNetV2 + FastAPI + React</p>
        </div>
      </footer>
    </div>
  )
}

export default App
