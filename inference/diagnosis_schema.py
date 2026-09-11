from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class CropInfo(BaseModel):
    name: str = Field(description="Detected crop family: cotton, rice, wheat, maize, sugarcane, or unknown")
    confidence: float = Field(description="Confidence score in crop identification [0.0, 1.0]")
    status: str = Field(default="UNKNOWN", description="VERIFIED, TENTATIVE, or UNKNOWN")


class PlantPartInfo(BaseModel):
    name: str = Field(description="Detected plant organ: leaf, panicle, stem, boll, root, or unknown")
    confidence: float = Field(description="Confidence score in plant organ identification [0.0, 1.0]")
    status: str = Field(default="UNKNOWN", description="DETECTED, UNCERTAIN, or UNKNOWN")


class PrimaryModelInfo(BaseModel):
    status: str = Field(description="Status of closed-set CNN: 'accepted', 'rejected', or 'bypassed'")
    prediction: Optional[str] = Field(default=None, description="Accepted disease prediction, or None if rejected")
    confidence: Optional[float] = Field(default=None, description="Confidence of accepted prediction")
    raw_top_prediction: str = Field(description="Raw top CNN class before compatibility gating (for debug telemetry)")
    raw_confidence: float = Field(description="Raw softmax confidence before compatibility gating")
    rejection_reason: Optional[str] = Field(default=None, description="Explicit reason if primary prediction was rejected")
    rejection_reasons: List[str] = Field(default_factory=list, description="Structured list of all gate rejection reasons")
    accepted: bool = Field(default=False, description="True ONLY if CNN passed all required safety gates")
    ood: bool = Field(default=False, description="True if image exceeded multi-metric OOD thresholds")
    crop_compatible: bool = Field(default=False, description="True if top prediction is compatible with detected crop")
    plant_part_compatible: bool = Field(default=False, description="True if top prediction is compatible with detected organ")
    gate_status: str = Field(default="REJECTED", description="'ACCEPTED' or 'REJECTED'")
    energy_score: float = Field(default=0.0)
    entropy: float = Field(default=0.0)
    top_candidates: List[Dict[str, Any]] = Field(default_factory=list)


class FallbackInfo(BaseModel):
    provider: str = Field(description="'gemini_vision' or 'unavailable'")
    status: str = Field(description="'SUCCESS', 'UNCERTAIN', 'UNKNOWN', 'INSUFFICIENT_EVIDENCE', 'UNAVAILABLE', or 'ERROR'")
    crop: str = Field(default="unknown", description="Crop identified by Gemini")
    plant_part: str = Field(default="unknown", description="Plant part identified by Gemini")
    diagnosis: str = Field(default="", description="Condition diagnosed by Gemini")
    assessment_strength: str = Field(default="LOW", description="'HIGH', 'MEDIUM', or 'LOW' visual assessment strength")
    model_reported_confidence: float = Field(default=0.0, description="Raw uncalibrated self-reported score from vision model")
    reasoning: Optional[str] = Field(default=None, description="Concise evidence summary without chain-of-thought")
    evidence: List[str] = Field(default_factory=list, description="Extracted visual symptom markers")


class SemanticComparisonInfo(BaseModel):
    crop_agreement: bool = Field(default=False)
    plant_part_agreement: bool = False
    disease_semantic_match: bool = False
    overall_agreement: bool = False
    match_level: str = "NONE"
    explanation: str = ""
    canonical_class: Optional[str] = None


class DiagnosisInfo(BaseModel):
    name: str = Field(description="Final condition name or 'Unable to determine disease reliably'")
    type: str = Field(description="'disease', 'pest', 'disorder', 'healthy', or 'unknown'")
    confidence: str = Field(description="'low', 'medium', or 'high'")
    status: str = Field(description="Resolution state: 'CONSENSUS', 'CNN_ONLY', 'GEMINI_SUSPECTED', 'CONFLICT', 'UNKNOWN', 'INSUFFICIENT_EVIDENCE'")
    consensus_state: Optional[str] = Field(default=None, description="Detailed consensus resolution")
    resolution: str = Field(default="UNKNOWN", description="CONSENSUS, CNN_ONLY, GEMINI_SUSPECTED, CONFLICT, UNKNOWN, INSUFFICIENT_EVIDENCE")
    source: str = Field(default="NONE", description="'CONSENSUS', 'CNN', 'GEMINI', 'DISAGREEMENT', or 'NONE'")
    validated_by_cnn: bool = Field(default=False, description="True ONLY if CNN safety gates passed and agreed")
    farmer_headline: str = Field(default="", description="Simple farmer-facing diagnosis headline")
    farmer_subheading: str = Field(default="", description="Farmer-facing trust and authority subtext")


class AdvisoryPlan(BaseModel):
    urgency: str
    disease_description: str
    chemical_control: List[str] = Field(default_factory=list)
    organic_control: List[str] = Field(default_factory=list)
    cultural_practices: List[str] = Field(default_factory=list)
    expert_verification_note: str


class StandardizedDiagnosisResponse(BaseModel):
    """
    Standardized, defensive diagnosis response conforming to hierarchical architecture.
    Strictly separates user-facing diagnosis from raw debug telemetry.
    """
    crop: CropInfo
    plant_part: PlantPartInfo
    primary_model: PrimaryModelInfo
    fallback: FallbackInfo
    diagnosis: DiagnosisInfo
    evidence: List[str]
    recommendation: str
    requires_expert_verification: bool
    advisory: AdvisoryPlan
    
    # Dual-model comparison matrix
    comparison: Optional[SemanticComparisonInfo] = None
    resolution: Optional[str] = None

    # Preserved multi-model telemetry for UI visualizers
    pests: Dict[str, Any] = Field(default_factory=dict)
    segmentation: Dict[str, Any] = Field(default_factory=dict)
    visualizations: Dict[str, str] = Field(default_factory=dict)
