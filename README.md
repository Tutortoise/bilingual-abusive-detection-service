# Bilingual Abusive Text Detection Engine

![Abusive Text Detection Engine](./architecture.jpg)

A content moderation engine designed to maintain professional standards in our tutor-finding platform. Supports both English and Indonesian languages.

## Overview

This engine provides enterprise-grade content moderation capabilities:

- Real-time detection of inappropriate content
- Multilingual support (English and Indonesian)
- High-precision text classification
- Scalable architecture for production deployment
- Production-ready API endpoints
- Comprehensive batch processing capabilities

## Key Features

### Content Analysis

- Multi-language support (EN/ID)
- Character substitution detection
- Context-aware classification
- Pattern recognition for evasion attempts
- Real-time content validation

### Technical Capabilities

- Low-latency response times (<100ms)
- High-throughput batch processing
- Scalable worker configuration
- Configurable confidence thresholds
- Comprehensive logging and monitoring

## Technical Stack

- **Runtime**: Python 3.11+
- **Framework**: FastAPI
- **Server**: Granian (High-performance ASGI server)
- **ML Framework**: TensorFlow 2.x
- **Package Management**: UV

## Installation

### Using Docker (Recommended)

```bash
# Build the image
docker build -t abusive-detection:latest .

# Run the container
docker run -d -p 8000:8000 abusive-detection:latest
```

### Manual Installation

```bash
# Install dependencies
uv sync

# Start the server
granian web.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --interface asgi \
    --workers $(nproc)
```

## Configuration

### Environment Variables

| Variable                   | Description          | Default   |
| -------------------------- | -------------------- | --------- |
| `GRANIAN_HOST`             | Server host          | `0.0.0.0` |
| `GRANIAN_PORT`             | Server port          | `8000`    |
| `GRANIAN_WORKERS_PER_CORE` | Workers per CPU core | `2`       |
| `GRANIAN_MAX_WORKERS`      | Maximum worker limit | `32`      |
| `GRANIAN_LOG_LEVEL`        | Logging verbosity    | `info`    |

## API Reference

### Single Text Analysis

```http
POST /predict
Content-Type: application/json

{
    "text": "Content to analyze"
}
```

### Batch Analysis

```http
POST /predict_batch
Content-Type: application/json

{
    "texts": [
        "First content to analyze",
        "Second content to analyze"
    ]
}
```

### Response Schema

```typescript
interface PredictionResponse {
  text: string;
  probability: float; // Range: 0-1
  is_abusive: boolean;
  confidence: float; // Range: 0-1
  early_detection: boolean;
  matched_words: string[];
}
```

### Example Response

```json
{
  "text": "Sample text for analysis",
  "probability": 0.12,
  "is_abusive": false,
  "confidence": 0.88,
  "early_detection": false,
  "matched_words": []
}
```

## Model Training

### Datasets

- **English**: [Hate Speech and Offensive Language Detection](https://www.kaggle.com/datasets/thedevastator/hate-speech-and-offensive-language-detection/data)
- **Indonesian**: [Indonesian Abusive and Hate Speech Twitter Text](https://www.kaggle.com/datasets/ilhamfp31/indonesian-abusive-and-hate-speech-twitter-text/data)

## Health Monitoring

```http
GET /health
```

Returns service health status and basic metrics.
