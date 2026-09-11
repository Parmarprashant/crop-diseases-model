from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PestDetectionItem(BaseModel):
    label: str
    confidence: float
    box: List[float]


class PestResult(BaseModel):
    count: int
    detections: List[PestDetectionItem] = []


class CandidatePrediction(BaseModel):
    class_name: str
    confidence: float


class DiagnosisDetails(BaseModel):
    disease_name: str
    confidence: float
    model_source: str
    crop: Optional[str] = "Field Crop"
    reasoning: Optional[str] = None
    primary_model_tentative: Optional[str] = None
    primary_model_confidence: Optional[float] = None
    top_candidates: List[CandidatePrediction] = []


class SegmentationResult(BaseModel):
    infected_area_pct: float
    severity_category: str
    leaf_pixels: int
    lesion_pixels: int


class AdvisoryResult(BaseModel):
    disease_identified: str
    urgency: str
    chemical_control: List[str] = []
    organic_control: List[str] = []
    cultural_practices: List[str] = []
    pest_alerts: Optional[List[str]] = []


class Visualizations(BaseModel):
    original: str
    attention_heatmap: str
    pest_detections: str
    lesion_segmentation: str


class DiagnosisResponse(BaseModel):
    decision_path: str = Field(..., description="'PRIMARY_MODEL_ACCEPTED' or 'GEMINI_VISION_FALLBACK'")
    is_fallback: bool
    confidence_threshold: float = 0.80
    diagnosis: DiagnosisDetails
    pests: PestResult
    segmentation: SegmentationResult
    advisory: AdvisoryResult
    visualizations: Visualizations


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    models_loaded: Dict[str, bool]
    confidence_threshold: float
