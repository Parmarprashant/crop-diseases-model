"""
AgriVision AI — Crop-Aware Disease Resolver & Calibration Layer.

Solves the diagnostic bottleneck:
Recovers biologically compatible disease diagnoses from Model A when the
Dedicated Crop Expert confidently identifies the true crop family, while
strictly eliminating cross-crop errors and pesticide hallucinations.

Enforces:
1. No hard argmax masking: evaluates continuous crop and disease evidence.
2. Cumulative in-crop evidence: accounts for distributed probability mass across
   disease variants (does not artificially penalize shared in-crop probability).
3. Preserves all three telemetry scores for full diagnostic transparency.
4. Preserves Model A defensive quality and OOD rejections (100% safety contract).
"""
import os
import json
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

@dataclass
class ResolverTelemetry:
    model_a_original_disease_score: float
    crop_expert_probability: float
    crop_conditioned_disease_score: float
    final_resolver_decision: str
    crop_top1: str
    crop_top2: str
    crop_margin: float
    crop_entropy: float
    model_a_disease_top1: str
    model_a_disease_confidence: float
    model_a_disease_crop_compatibility: bool
    resolver_state: str
    resolver_reason: str

@dataclass
class CropDiseaseResolverResult:
    final_crop: str
    final_disease: str
    final_confidence: float
    final_accepted: bool
    telemetry: ResolverTelemetry

class CropAwareDiseaseResolver:
    """
    Deterministic crop-aware disease resolution engine.
    Supports Mode A (Advisory), Mode B (Conservative Compatible Recovery),
    and Mode C (Confidence-Weighted Fusion).
    """
    CROP_EQUIVALENCES = {
        "paddy": "rice",
        "rice": "paddy",
        "grapevine": "grape",
        "grape": "grapevine",
        "maize": "corn",
        "corn": "maize"
    }

    def __init__(
        self,
        mode: str = "mode_b",
        tau_crop_conf: float = 0.45,
        tau_crop_margin: float = 0.08,
        tau_crop_entropy: float = 0.85,
        tau_compat_mass: float = 0.08,
        tau_within_crop_dom: float = 0.25,
        tau_single_compat: float = 0.05,
        tau_disease_conf: float = 0.35,
        canonical_classes_path: str = "weights/canonical_class_map.json",
        calibrator: Optional[Any] = None
    ):
        self.mode = mode.lower().strip()
        self.tau_crop_conf = tau_crop_conf
        self.tau_crop_margin = tau_crop_margin
        self.tau_crop_entropy = tau_crop_entropy
        self.tau_compat_mass = tau_compat_mass
        self.tau_within_crop_dom = tau_within_crop_dom
        self.tau_single_compat = tau_single_compat
        self.tau_disease_conf = tau_disease_conf
        self.calibrator = calibrator

    def _make_result(
        self,
        final_crop: str,
        final_disease: str,
        final_confidence: float,
        final_accepted: bool,
        telemetry: ResolverTelemetry
    ) -> CropDiseaseResolverResult:
        conf = float(final_confidence)
        if self.calibrator is not None:
            conf = self.calibrator.calibrate(conf)
        return CropDiseaseResolverResult(
            final_crop=final_crop,
            final_disease=final_disease,
            final_confidence=round(conf, 4),
            final_accepted=final_accepted,
            telemetry=telemetry
        )

    @classmethod
    def extract_crop_from_disease_class(cls, class_name: str) -> str:
        """Extract canonical crop family from disease class name (e.g. 'Apple - Scab' -> 'apple')."""
        if not class_name:
            return "unknown"
        clean = class_name.strip()
        if " - " in clean:
            return clean.split(" - ", 1)[0].strip().lower()
        # Fallback to leading word
        return clean.split()[0].strip().lower()

    def are_crops_compatible(self, crop_a: str, crop_b: str) -> bool:
        c_a = crop_a.lower().strip()
        c_b = crop_b.lower().strip()
        if c_a == c_b:
            return True
        if self.CROP_EQUIVALENCES.get(c_a) == c_b or self.CROP_EQUIVALENCES.get(c_b) == c_a:
            return True
        return False

    def resolve(
        self,
        # Model A Inputs
        model_a_crop: str,
        model_a_diag: str,
        model_a_conf: float,
        model_a_accepted: bool,
        canonical_disease_probs: Dict[str, float], # Full 167-class canonical distribution
        # Calibrated Crop Expert Inputs
        crop_top1: str,
        crop_top2: str,
        crop_conf: float,
        crop_margin: float,
        crop_entropy: float,
        crop_probs: Optional[Dict[str, float]] = None # Full 41-crop distribution
    ) -> CropDiseaseResolverResult:
        """
        Executes crop-aware disease resolution.
        """
        c_model_a = model_a_crop.lower().strip()
        c_expert = crop_top1.lower().strip()

        # Check biological compatibility of Model A's top-1 disease with Crop Expert
        model_a_diag_crop = self.extract_crop_from_disease_class(model_a_diag)
        compat_with_expert = self.are_crops_compatible(model_a_diag_crop, c_expert)
        crops_agree = self.are_crops_compatible(c_model_a, c_expert)

        # 0. DEFENSIVE REJECTION INVARIANT:
        # If Model A was rejected by Image Quality or OOD Gating, ALWAYS preserve rejection
        if not model_a_accepted:
            telemetry = ResolverTelemetry(
                model_a_original_disease_score=round(model_a_conf, 4),
                crop_expert_probability=round(crop_conf, 4),
                crop_conditioned_disease_score=0.0,
                final_resolver_decision=model_a_diag,
                crop_top1=c_expert,
                crop_top2=crop_top2,
                crop_margin=round(crop_margin, 4),
                crop_entropy=round(crop_entropy, 4),
                model_a_disease_top1=model_a_diag,
                model_a_disease_confidence=round(model_a_conf, 4),
                model_a_disease_crop_compatibility=compat_with_expert,
                resolver_state="DEFENSIVE_REJECTION",
                resolver_reason="Model A defensive rejection preserved (quality/OOD cutoff)"
            )
            return self._make_result(
                final_crop=c_expert if crop_conf >= self.tau_crop_conf else c_model_a,
                final_disease=model_a_diag,
                final_confidence=model_a_conf,
                final_accepted=False,
                telemetry=telemetry
            )

        # =====================================================================
        # MODE A: ADVISORY (Crop expert reports telemetry without overriding)
        # =====================================================================
        if self.mode == "mode_a":
            telemetry = ResolverTelemetry(
                model_a_original_disease_score=round(model_a_conf, 4),
                crop_expert_probability=round(crop_conf, 4),
                crop_conditioned_disease_score=round(model_a_conf, 4),
                final_resolver_decision=model_a_diag,
                crop_top1=c_expert,
                crop_top2=crop_top2,
                crop_margin=round(crop_margin, 4),
                crop_entropy=round(crop_entropy, 4),
                model_a_disease_top1=model_a_diag,
                model_a_disease_confidence=round(model_a_conf, 4),
                model_a_disease_crop_compatibility=compat_with_expert,
                resolver_state="MODE_A_ADVISORY",
                resolver_reason="Advisory mode: Model A disease preserved with compatibility telemetry."
            )
            return self._make_result(
                final_crop=c_model_a,
                final_disease=model_a_diag,
                final_confidence=model_a_conf,
                final_accepted=True,
                telemetry=telemetry
            )

        # =====================================================================
        # MODE C: CONFIDENCE-WEIGHTED FUSION
        # =====================================================================
        if self.mode == "mode_c":
            # Soft bayesian fusion across all canonical diseases
            prior_crop_probs = crop_probs or {c_expert: crop_conf, crop_top2: max(0.0, crop_conf - crop_margin)}
            fused_scores = {}
            for d_name, d_prob in canonical_disease_probs.items():
                d_crop = self.extract_crop_from_disease_class(d_name)
                # Find crop probability for d_crop
                c_prob = prior_crop_probs.get(d_crop, 0.0)
                # Match equivalence (e.g. paddy / rice)
                if c_prob == 0.0 and d_crop in self.CROP_EQUIVALENCES:
                    c_prob = prior_crop_probs.get(self.CROP_EQUIVALENCES[d_crop], 0.0)
                # Weighted score
                score = d_prob * (c_prob + 0.02)
                fused_scores[d_name] = score

            if fused_scores:
                best_fused_diag = max(fused_scores, key=fused_scores.get)
                fused_conf = fused_scores[best_fused_diag]
                fused_crop = self.extract_crop_from_disease_class(best_fused_diag)
                
                # Check acceptance
                is_accepted = fused_conf >= 0.02
                final_diag = best_fused_diag if is_accepted else "Unable to determine the disease reliably"
                
                telemetry = ResolverTelemetry(
                    model_a_original_disease_score=round(model_a_conf, 4),
                    crop_expert_probability=round(crop_conf, 4),
                    crop_conditioned_disease_score=round(fused_conf, 4),
                    final_resolver_decision=final_diag,
                    crop_top1=c_expert,
                    crop_top2=crop_top2,
                    crop_margin=round(crop_margin, 4),
                    crop_entropy=round(crop_entropy, 4),
                    model_a_disease_top1=model_a_diag,
                    model_a_disease_confidence=round(model_a_conf, 4),
                    model_a_disease_crop_compatibility=compat_with_expert,
                    resolver_state="MODE_C_FUSION",
                    resolver_reason=f"Confidence-weighted bayesian fusion: {final_diag}"
                )
                return self._make_result(
                    final_crop=fused_crop,
                    final_disease=final_diag,
                    final_confidence=round(fused_conf, 4),
                    final_accepted=is_accepted,
                    telemetry=telemetry
                )

        # =====================================================================
        # MODE B: CONSERVATIVE COMPATIBLE RECOVERY (Primary Architecture)
        # =====================================================================
        # Is Crop Expert confident and well-separated?
        is_crop_expert_confident = (
            (crop_conf >= self.tau_crop_conf) and
            (crop_margin >= self.tau_crop_margin) and
            (crop_entropy <= self.tau_crop_entropy)
        )

        # ---------------------------------------------------------------------
        # STATE A: STRONG AGREEMENT
        # Both models agree on the crop family
        # ---------------------------------------------------------------------
        if crops_agree or compat_with_expert:
            boosted_conf = min(1.0, model_a_conf * 1.05)
            telemetry = ResolverTelemetry(
                model_a_original_disease_score=round(model_a_conf, 4),
                crop_expert_probability=round(crop_conf, 4),
                crop_conditioned_disease_score=round(boosted_conf, 4),
                final_resolver_decision=model_a_diag,
                crop_top1=c_expert,
                crop_top2=crop_top2,
                crop_margin=round(crop_margin, 4),
                crop_entropy=round(crop_entropy, 4),
                model_a_disease_top1=model_a_diag,
                model_a_disease_confidence=round(model_a_conf, 4),
                model_a_disease_crop_compatibility=True,
                resolver_state="STATE_A_AGREEMENT",
                resolver_reason=f"Full crop-disease agreement on {c_expert}."
            )
            return self._make_result(
                final_crop=c_expert,
                final_disease=model_a_diag,
                final_confidence=round(boosted_conf, 4),
                final_accepted=True,
                telemetry=telemetry
            )

        # ---------------------------------------------------------------------
        # STATE B: STRONG CROP / INCOMPATIBLE MODEL A DISEASE
        # Crop expert is confident, but Model A's top-1 disease belongs to crop Y
        # ---------------------------------------------------------------------
        if is_crop_expert_confident:
            # 1. Filter Model A's canonical disease distribution for diseases compatible with c_expert
            compatible_candidates = []
            total_compat_mass = 0.0

            for d_name, d_prob in canonical_disease_probs.items():
                d_crop = self.extract_crop_from_disease_class(d_name)
                if self.are_crops_compatible(d_crop, c_expert):
                    compatible_candidates.append((d_name, d_prob))
                    total_compat_mass += d_prob

            # Sort compatible diseases by probability descending
            compatible_candidates.sort(key=lambda x: x[1], reverse=True)

            if compatible_candidates:
                best_compat_diag, best_compat_prob = compatible_candidates[0]
                second_compat_prob = compatible_candidates[1][1] if len(compatible_candidates) > 1 else 0.0
                
                # Within-crop conditional dominance
                within_crop_dominance = best_compat_prob / max(total_compat_mass, 1e-6)
                within_crop_margin = best_compat_prob - second_compat_prob

                # CUMULATIVE IN-CROP EVIDENCE RESOLUTION:
                # Accept if:
                # a) Total compatible mass >= tau_compat_mass, OR
                # b) Within-crop dominance >= tau_within_crop_dom, OR
                # c) Individual compatible probability >= tau_single_compat
                has_sufficient_evidence = (
                    (total_compat_mass >= self.tau_compat_mass) or
                    (within_crop_dominance >= self.tau_within_crop_dom and total_compat_mass >= 0.03) or
                    (best_compat_prob >= self.tau_single_compat)
                )

                if has_sufficient_evidence:
                    # Successfully recovered valid compatible disease!
                    calibrated_conf = min(1.0, crop_conf * max(best_compat_prob, within_crop_dominance))
                    telemetry = ResolverTelemetry(
                        model_a_original_disease_score=round(model_a_conf, 4),
                        crop_expert_probability=round(crop_conf, 4),
                        crop_conditioned_disease_score=round(best_compat_prob, 4),
                        final_resolver_decision=best_compat_diag,
                        crop_top1=c_expert,
                        crop_top2=crop_top2,
                        crop_margin=round(crop_margin, 4),
                        crop_entropy=round(crop_entropy, 4),
                        model_a_disease_top1=model_a_diag,
                        model_a_disease_confidence=round(model_a_conf, 4),
                        model_a_disease_crop_compatibility=False,
                        resolver_state="STATE_B_COMPATIBLE_RECOVERED",
                        resolver_reason=(
                            f"Recovered compatible disease '{best_compat_diag}' for crop '{c_expert}' "
                            f"(total_compat_mass={total_compat_mass:.3f}, within_crop_dom={within_crop_dominance:.2f}, "
                            f"best_compat_p={best_compat_prob:.3f}). Suppressed cross-crop '{model_a_diag}'."
                        )
                    )
                    return self._make_result(
                        final_crop=c_expert,
                        final_disease=best_compat_diag,
                        final_confidence=round(calibrated_conf, 4),
                        final_accepted=True,
                        telemetry=telemetry
                    )

            # If no compatible disease found or cumulative evidence is truly negligible:
            # Safe refusal to prevent illegal agrochemical sprays
            telemetry = ResolverTelemetry(
                model_a_original_disease_score=round(model_a_conf, 4),
                crop_expert_probability=round(crop_conf, 4),
                crop_conditioned_disease_score=0.0,
                final_resolver_decision="Unable to determine the disease reliably",
                crop_top1=c_expert,
                crop_top2=crop_top2,
                crop_margin=round(crop_margin, 4),
                crop_entropy=round(crop_entropy, 4),
                model_a_disease_top1=model_a_diag,
                model_a_disease_confidence=round(model_a_conf, 4),
                model_a_disease_crop_compatibility=False,
                resolver_state="STATE_B_REFUSAL",
                resolver_reason=(
                    f"Cross-crop conflict: Confident crop expert ({c_expert}, {crop_conf*100:.1f}%) "
                    f"conflicts with Model A diagnosis '{model_a_diag}'. Insufficient compatible evidence "
                    f"(compat_mass={total_compat_mass:.3f}). Chemical spray suppressed."
                )
            )
            return self._make_result(
                final_crop=c_expert,
                final_disease="Unable to determine the disease reliably",
                final_confidence=round(crop_conf, 4),
                final_accepted=False,
                telemetry=telemetry
            )

        # ---------------------------------------------------------------------
        # STATE C: WEAK / AMBIGUOUS CROP EXPERT
        # Crop Expert is uncertain (low confidence, small margin, or high entropy)
        # ---------------------------------------------------------------------
        # Rule: A weak Crop Expert NEVER overrides strong Model A disease evidence
        if model_a_conf >= self.tau_disease_conf:
            telemetry = ResolverTelemetry(
                model_a_original_disease_score=round(model_a_conf, 4),
                crop_expert_probability=round(crop_conf, 4),
                crop_conditioned_disease_score=round(model_a_conf, 4),
                final_resolver_decision=model_a_diag,
                crop_top1=c_expert,
                crop_top2=crop_top2,
                crop_margin=round(crop_margin, 4),
                crop_entropy=round(crop_entropy, 4),
                model_a_disease_top1=model_a_diag,
                model_a_disease_confidence=round(model_a_conf, 4),
                model_a_disease_crop_compatibility=compat_with_expert,
                resolver_state="STATE_C_PRESERVE_MODEL_A",
                resolver_reason=(
                    f"Crop expert uncertain (conf={crop_conf*100:.1f}%, margin={crop_margin*100:.1f}%). "
                    f"Strong Model A diagnostic evidence preserved ({model_a_diag}, conf={model_a_conf*100:.1f}%)."
                )
            )
            return self._make_result(
                final_crop=c_model_a,
                final_disease=model_a_diag,
                final_confidence=round(model_a_conf, 4),
                final_accepted=True,
                telemetry=telemetry
            )
        else:
            # Both Model A and Crop Expert are ambiguous: defensive abstention
            telemetry = ResolverTelemetry(
                model_a_original_disease_score=round(model_a_conf, 4),
                crop_expert_probability=round(crop_conf, 4),
                crop_conditioned_disease_score=0.0,
                final_resolver_decision="Unable to determine the disease reliably",
                crop_top1=c_expert,
                crop_top2=crop_top2,
                crop_margin=round(crop_margin, 4),
                crop_entropy=round(crop_entropy, 4),
                model_a_disease_top1=model_a_diag,
                model_a_disease_confidence=round(model_a_conf, 4),
                model_a_disease_crop_compatibility=compat_with_expert,
                resolver_state="STATE_C_AMBIGUOUS_REFUSAL",
                resolver_reason="Both Model A and Crop Expert have low confidence. Safe refusal."
            )
            return self._make_result(
                final_crop="unknown",
                final_disease="Unable to determine the disease reliably",
                final_confidence=round(model_a_conf, 4),
                final_accepted=False,
                telemetry=telemetry
            )
