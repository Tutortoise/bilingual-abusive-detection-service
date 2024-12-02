from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, validator
from typing import List
from .model import AbusiveTextDetector
from .config import MODEL_PATH, VECTORIZER_CONFIG_PATH
from .exceptions import EmptyTextError


class TextRequest(BaseModel):
    text: str

    @validator("text")
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v


class BatchTextRequest(BaseModel):
    texts: List[str]

    @validator("texts")
    def texts_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("Texts list cannot be empty")
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty")
        return v


class PredictionResponse(BaseModel):
    text: str
    probability: float
    is_abusive: bool
    confidence: float
    early_detection: bool
    matched_words: List[str] = []


class ErrorResponse(BaseModel):
    detail: str


app = FastAPI(
    title="Abusive Text Detection API",
    description="API for detecting abusive text in English and Indonesian",
    version="1.0.0",
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

detector = AbusiveTextDetector(
    model_path=MODEL_PATH, vectorizer_config_path=VECTORIZER_CONFIG_PATH, threshold=0.7
)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Empty text or invalid input",
        }
    },
)
async def predict_single(request: TextRequest):
    try:
        result = detector.predict(request.text)
        return result
    except EmptyTextError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict_batch",
    response_model=List[PredictionResponse],
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Empty text or invalid input in batch",
        }
    },
)
async def predict_batch(request: BatchTextRequest):
    try:
        results = detector.predict_batch(request.texts)
        return results
    except EmptyTextError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
