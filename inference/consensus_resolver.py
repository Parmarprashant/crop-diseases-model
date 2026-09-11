from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from inference.crop_taxonomy import SemanticDiagnosisMatcher, get_class_metadata, CLASS_TAXONOMY


@dataclass
class ConsensusResult:
    consensus_state: str         # "CONSENSUS", "CNN_ONLY", "GEMINI_SUSPECTED", "CONFLICT", "UNKNOWN", "INSUFFICIENT_EVIDENCE"
    resolution: str              # Canonical resolution identifier (same as consensus_state)
    source: str                  # "CONSENSUS", "CNN", "GEMINI", "DISAGREEMENT", "NONE"
    validated_by_cnn: bool       # True ONLY if CNN passed all safety gates and validated the diagnosis
    final_diagnosis_name: str
    final_diagnosis_type: str    # "disease", "pest", "disorder", "healthy", "unknown"
    confidence: str              # "low", "medium", "high"
    status: str                  # Resolution string
    explanation: str
    recommendation: str
    farmer_headline: str
    farmer_subheading: str
    requires_expert_verification: bool
    chemical_control: List[str]
    organic_control: List[str]
    cultural_practices: List[str]
    final_crop: str = "unknown"
    final_crop_confidence: float = 0.0
    final_crop_status: str = "UNKNOWN"
    final_plant_part: str = "unknown"
    final_plant_part_confidence: float = 0.0
    final_plant_part_status: str = "UNKNOWN"
    comparison_telemetry: Dict[str, Any] = field(default_factory=dict)


class ConsensusResolver:
    """
    Centralized Dual-Model Consensus Resolver.
    
    CRITICAL ARCHITECTURAL MANDATES:
    1. AGREEMENT != VALIDATION:
       A raw CNN prediction must NEVER become a final diagnosis if the CNN failed
       any required safety gate.
       If CNN is REJECTED and Gemini independently says the same disease:
       -> Resolution is GEMINI_SUSPECTED (NOT CONSENSUS, validated_by_cnn=False).
    2. Conservative Semantic Matching:
       Exact taxonomy match -> Approved alias match -> Otherwise: NO MATCH.
    3. Structural Invariant:
       If CNN was rejected, validated_by_cnn is structurally FALSE.
    4. Advisory Safety:
       Disease-specific chemical sprays are permitted ONLY for CONSENSUS or CNN_ONLY.
       Strictly suppressed for GEMINI_SUSPECTED, CONFLICT, UNKNOWN, INSUFFICIENT_EVIDENCE.
    """

    @staticmethod
    def resolve(
        crop: str,
        plant_part: str,
        primary_status: str,                    # "accepted", "rejected", "bypassed"
        primary_prediction: Optional[str],       # e.g. "Cotton Anthracnose" (None if rejected)
        raw_top_prediction: str,                 # for debug telemetry / context
        raw_confidence: float,
        primary_rejection_reasons: List[str],
        fallback_provider: str,                  # "gemini_vision", "unavailable", "not_needed"
        fallback_status: str,                    # "invoked", "unavailable", "not_needed", "error"
        gemini_result: Optional[Dict[str, Any]] = None
    ) -> ConsensusResult:
        crop_clean = crop.lower().strip() if crop else "unknown"
        part_clean = plant_part.lower().strip() if plant_part else "unknown"
        cnn_valid = (primary_status == "accepted" and bool(primary_prediction))

        # Extract Gemini assessment
        gemini_invoked = (fallback_status == "invoked" and gemini_result is not None)
        default_status = "SUCCESS" if (gemini_result and (gemini_result.get("diagnosis") or gemini_result.get("disease_name"))) else "UNAVAILABLE"
        gemini_status = (gemini_result.get("status") or default_status).upper() if gemini_invoked else "UNAVAILABLE"
        gemini_disease = (gemini_result.get("diagnosis") or gemini_result.get("disease_name") or "").strip() if gemini_invoked else ""
        gemini_strength = str(gemini_result.get("assessment_strength", "LOW")).upper() if gemini_invoked else "LOW"
        gemini_model_conf = float(gemini_result.get("model_reported_confidence", gemini_result.get("confidence", 0.0))) if gemini_invoked else 0.0
        gemini_crop = (gemini_result.get("crop") or "").lower().strip() if gemini_invoked else ""
        gemini_part = (gemini_result.get("plant_part") or "").lower().strip() if gemini_invoked else ""
        gemini_reasoning = (gemini_result.get("reasoning_summary") or gemini_result.get("visual_reasoning") or "") if gemini_invoked else ""
        gemini_evidence = gemini_result.get("evidence", []) if gemini_invoked else []

        gemini_is_unknown = (
            not gemini_disease or
            gemini_disease.lower() in ["unknown", "unknown condition", "insufficient evidence", "none", "error", "unavailable"] or
            gemini_status in ["UNAVAILABLE", "ERROR", "UNKNOWN", "INSUFFICIENT_EVIDENCE"]
        )
        gemini_valid = gemini_invoked and not gemini_is_unknown

        # Semantic Comparison
        is_semantic_match = False
        match_explanation = "Comparison not applicable"
        comparison_telemetry = {
            "crop_agreement": False,
            "plant_part_agreement": False,
            "disease_semantic_match": False,
            "overall_agreement": False,
            "match_level": "NONE",
            "explanation": match_explanation,
            "canonical_class": None
        }

        # Compare active CNN prediction (or raw CNN if rejected) against Gemini
        comparison_target_disease = primary_prediction if cnn_valid else raw_top_prediction
        if comparison_target_disease and gemini_disease:
            is_semantic_match, match_explanation, comp_data = SemanticDiagnosisMatcher.match(
                crop1=crop_clean,
                plant_part1=part_clean,
                diagnosis1=comparison_target_disease,
                crop2=gemini_crop or crop_clean,
                plant_part2=gemini_part or part_clean,
                diagnosis2=gemini_disease
            )
            comparison_telemetry.update(comp_data)
            comparison_telemetry["explanation"] = match_explanation
            comparison_telemetry["disease_semantic_match"] = is_semantic_match
            comparison_telemetry["overall_agreement"] = is_semantic_match and comp_data.get("crop_match", False)

        # ---------------------------------------------------------------------
        # CASE 1: CNN ACCEPTED + GEMINI VALID (SAME -> CONSENSUS | DIFF -> CONFLICT)
        # ---------------------------------------------------------------------
        if cnn_valid and gemini_valid:
            if is_semantic_match:
                # Resolution: CONSENSUS
                final_name = primary_prediction or gemini_disease
                final_crop = gemini_crop if (gemini_crop and gemini_crop != "unknown") else crop_clean
                final_part = gemini_part if (gemini_part and gemini_part != "unknown") else part_clean
                return ConsensusResult(
                    consensus_state="CONSENSUS",
                    resolution="CONSENSUS",
                    source="CONSENSUS",
                    validated_by_cnn=True,
                    final_diagnosis_name=final_name,
                    final_diagnosis_type="disease",
                    confidence="high" if raw_confidence >= 0.80 and gemini_strength == "HIGH" else "medium",
                    status="CONSENSUS",
                    explanation=(
                        f"Independent Model Consensus: Primary CNN ({raw_confidence*100:.1f}%) and "
                        f"Gemini Vision independently agree on '{final_name}' ({match_explanation}). "
                        f"Domain compatibility, plant-part gating, and OOD calibration verified."
                    ),
                    recommendation="Both independent visual assessments agree. Follow standard integrated management below.",
                    farmer_headline=final_name,
                    farmer_subheading="AI consensus: CNN + Gemini. Both models independently agree.",
                    requires_expert_verification=False,
                    chemical_control=[],  # Downstream advisory engine will populate from taxonomy
                    organic_control=[],
                    cultural_practices=[],
                    final_crop=final_crop,
                    final_crop_confidence=max(raw_confidence, gemini_model_conf),
                    final_crop_status="VERIFIED",
                    final_plant_part=final_part,
                    final_plant_part_confidence=0.90,
                    final_plant_part_status="DETECTED",
                    comparison_telemetry=comparison_telemetry
                )
            else:
                # Resolution: CONFLICT
                final_name = f"Diagnosis Conflict ({primary_prediction} vs {gemini_disease})"
                return ConsensusResult(
                    consensus_state="CONFLICT",
                    resolution="CONFLICT",
                    source="DISAGREEMENT",
                    validated_by_cnn=False,
                    final_diagnosis_name=final_name,
                    final_diagnosis_type="disorder",
                    confidence="low",
                    status="CONFLICT",
                    explanation=(
                        f"Independent Model Disagreement: Primary CNN identified '{primary_prediction}' ({raw_confidence*100:.1f}%), "
                        f"while Gemini Vision independently identified '{gemini_disease}'. "
                        f"Neither model is silently chosen."
                    ),
                    recommendation="Obtain field examination or agricultural extension verification before applying treatments.",
                    farmer_headline="AI assessments disagree",
                    farmer_subheading="Expert verification recommended",
                    requires_expert_verification=True,
                    chemical_control=[],  # STRICTLY SUPPRESSED on conflict
                    organic_control=[],
                    cultural_practices=[
                        f"Physically inspect plants to differentiate '{primary_prediction}' from '{gemini_disease}'.",
                        "Consult local agricultural extension officer before applying chemical sprays."
                    ],
                    final_crop=crop_clean,
                    final_crop_confidence=raw_confidence,
                    final_crop_status="TENTATIVE",
                    final_plant_part=part_clean,
                    final_plant_part_confidence=0.50,
                    final_plant_part_status="UNCERTAIN",
                    comparison_telemetry=comparison_telemetry
                )

        # ---------------------------------------------------------------------
        # CASE 2: CNN ACCEPTED + GEMINI UNAVAILABLE -> CNN_ONLY
        # ---------------------------------------------------------------------
        if cnn_valid and not gemini_valid:
            final_name = primary_prediction or "Unknown Condition"
            return ConsensusResult(
                consensus_state="CNN_ONLY",
                resolution="CNN_ONLY",
                source="CNN",
                validated_by_cnn=True,
                final_diagnosis_name=final_name,
                final_diagnosis_type="disease",
                confidence="high" if raw_confidence >= 0.80 else "medium",
                status="CNN_ONLY",
                explanation=(
                    f"Primary CNN Accepted: Condition '{final_name}' ({raw_confidence*100:.1f}%) "
                    f"passed crop compatibility, plant-part gating, and energy-based OOD validation. "
                    f"Secondary vision model was unavailable."
                ),
                recommendation="Follow standard agronomic integrated disease management practices.",
                farmer_headline=final_name,
                farmer_subheading="Primary AI assessment: CNN. Gemini unavailable.",
                requires_expert_verification=False,
                chemical_control=[],  # Downstream advisory engine will populate from taxonomy
                organic_control=[],
                cultural_practices=[],
                final_crop=crop_clean,
                final_crop_confidence=raw_confidence,
                final_crop_status="VERIFIED",
                final_plant_part=part_clean,
                final_plant_part_confidence=0.85,
                final_plant_part_status="DETECTED",
                comparison_telemetry=comparison_telemetry
            )

        # ---------------------------------------------------------------------
        # CASE 3: CNN REJECTED + GEMINI VALID -> GEMINI_SUSPECTED
        # (CRITICAL: Even if Gemini agrees with the rejected CNN, this MUST be GEMINI_SUSPECTED!)
        # ---------------------------------------------------------------------
        if not cnn_valid and gemini_valid:
            resolved_crop = gemini_crop if (gemini_crop and gemini_crop != "unknown") else crop_clean
            resolved_part = gemini_part if (gemini_part and gemini_part != "unknown") else part_clean

            agreement_note = ""
            if is_semantic_match:
                agreement_note = (
                    f" Note: Primary CNN also indicated '{raw_top_prediction}' ({raw_confidence*100:.1f}%), "
                    f"but CNN failed safety gating ({'; '.join(primary_rejection_reasons)}) and is NOT validated."
                )

            return ConsensusResult(
                consensus_state="GEMINI_SUSPECTED",
                resolution="GEMINI_SUSPECTED",
                source="GEMINI",
                validated_by_cnn=False,  # STRUCTURAL INVARIANT: NEVER TRUE WHEN CNN IS REJECTED
                final_diagnosis_name=gemini_disease,
                final_diagnosis_type="disease",
                confidence="medium" if gemini_strength in ["HIGH", "MEDIUM"] else "low",
                status="GEMINI_SUSPECTED",
                explanation=(
                    f"Secondary Vision Assessment: Gemini Vision suspects '{gemini_disease}' on "
                    f"{resolved_crop.capitalize()} ({resolved_part.capitalize()}). "
                    f"Primary closed-set CNN was rejected by safety gates ({'; '.join(primary_rejection_reasons)}).{agreement_note}"
                ),
                recommendation="Consult an agronomist or extension specialist to confirm symptoms before initiating targeted treatments.",
                farmer_headline=f"Suspected: {gemini_disease}",
                farmer_subheading="AI visual assessment: Gemini. Expert verification recommended.",
                requires_expert_verification=True,
                chemical_control=[],  # STRICTLY SUPPRESSED for suspected conditions
                organic_control=[],   # Strictly no disease-specific treatments
                cultural_practices=[
                    f"Physically examine {resolved_crop.capitalize()} {resolved_part.capitalize()} for characteristic lesions of {gemini_disease}.",
                    "Seek physical confirmation from an agricultural extension officer before applying chemical sprays.",
                    "Capture a sharp, well-lit macro photo of the lesion margins under uniform lighting."
                ],
                final_crop=resolved_crop,
                final_crop_confidence=gemini_model_conf if gemini_model_conf > 0 else 0.70,
                final_crop_status="IDENTIFIED_BY_GEMINI" if (gemini_crop and gemini_crop != "unknown") else "TENTATIVE",
                final_plant_part=resolved_part,
                final_plant_part_confidence=0.75 if (gemini_part and gemini_part != "unknown") else 0.50,
                final_plant_part_status="DETECTED" if (gemini_part and gemini_part != "unknown") else "UNCERTAIN",
                comparison_telemetry=comparison_telemetry
            )

        # ---------------------------------------------------------------------
        # CASE 4 & 5: CNN REJECTED + GEMINI UNAVAILABLE / UNCERTAIN
        # -> INSUFFICIENT_EVIDENCE
        # ---------------------------------------------------------------------
        # Tailored agronomic explanation based on crop and organ
        if crop_clean == "rice" and part_clean in ["panicle", "ear", "grain", "head"]:
            custom_message = (
                "The image appears to show a rice panicle, but the primary disease model only supports "
                "leaf diseases and the secondary vision model is unavailable. "
                "Please provide a clearer image or obtain expert verification."
            )
        elif crop_clean != "unknown" and part_clean != "unknown":
            custom_message = (
                f"The image appears to show a {crop_clean} {part_clean}, but the primary disease model "
                f"does not support {part_clean} diseases ({'; '.join(primary_rejection_reasons)}) and the "
                f"secondary vision model is unavailable. Diagnostic certainty cannot be established."
            )
        else:
            custom_message = (
                f"The image exhibits out-of-distribution characteristics ({'; '.join(primary_rejection_reasons)}) "
                f"and the secondary vision model is unavailable. Diagnostic certainty cannot be established."
            )

        return ConsensusResult(
            consensus_state="INSUFFICIENT_EVIDENCE",
            resolution="INSUFFICIENT_EVIDENCE",
            source="NONE",
            validated_by_cnn=False,
            final_diagnosis_name="Unable to determine the disease reliably",
            final_diagnosis_type="unknown",
            confidence="low",
            status="INSUFFICIENT_EVIDENCE",
            explanation=custom_message,
            recommendation="Capture a sharp, well-lit photograph or submit a foliage sample to an agricultural extension officer.",
            farmer_headline="Unable to determine the disease reliably",
            farmer_subheading="Please provide a clearer image showing the affected plant part.",
            requires_expert_verification=True,
            chemical_control=[],  # STRICTLY SUPPRESSED
            organic_control=[],   # STRICTLY SUPPRESSED
            cultural_practices=[
                "Do not apply targeted chemical pesticides without a confirmed diagnosis.",
                "Inspect physical plants for specific symptoms (lesions, spore masses, entry holes).",
                "Ensure balanced plant nutrition and avoid excess nitrogen fertilization."
            ],
            final_crop=crop_clean,
            final_crop_confidence=raw_confidence,
            final_crop_status="TENTATIVE",
            final_plant_part=part_clean,
            final_plant_part_confidence=0.50,
            final_plant_part_status="UNCERTAIN",
            comparison_telemetry=comparison_telemetry
        )

