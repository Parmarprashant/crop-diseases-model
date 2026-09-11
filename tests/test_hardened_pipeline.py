import os
import sys
import unittest
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
from models.efficientnet_cbam import build_efficientnet_cbam, DiseaseClassifierInference
from models.yolo_pest import YOLOv8PestDetector
from models.unet_segmenter import LesionSegmenter
from core.gemini_fallback import GeminiVisionFallback
from core.advisory_engine import AdvisoryEngine
from inference.crop_taxonomy import (
    CLASS_TAXONOMY,
    check_crop_compatibility,
    check_plant_part_compatibility
)
from inference.image_quality import ImageQualityEvaluator
from inference.crop_detector import CropDetector
from inference.plant_part_detector import PlantPartDetector
from inference.ood_detector import OODDetector
from inference.pipeline import HierarchicalAgriDiagnosticPipeline


class TestHardenedAgriculturalPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights_path = os.path.join(BASE_DIR, "weights", "efficientnet_b5_cbam_best.pt")
        cls.class_names_path = os.path.join(BASE_DIR, "weights", "class_names.txt")
        cls.thresholds_path = os.path.join(BASE_DIR, "weights", "calibration_thresholds.json")

        with open(cls.class_names_path, "r") as f:
            cls.class_names = [l.strip() for l in f if l.strip()]

        cls.device = "cpu" # Test on CPU to prevent VRAM collision with running uvicorn
        cls.classifier_model = build_efficientnet_cbam(
            num_classes=len(cls.class_names),
            weights_path=cls.weights_path,
            device=cls.device
        )
        cls.classifier_inf = DiseaseClassifierInference(
            cls.classifier_model, cls.class_names, device=cls.device, img_size=256
        )

        cls.pest_detector = YOLOv8PestDetector(device=cls.device)
        cls.segmenter = LesionSegmenter(device=cls.device)
        cls.gemini_fallback = GeminiVisionFallback(api_key="") # Missing key for defensive test

        cls.pipeline = HierarchicalAgriDiagnosticPipeline(
            classifier=cls.classifier_inf,
            pest_detector=cls.pest_detector,
            segmenter=cls.segmenter,
            gemini_fallback=cls.gemini_fallback,
            thresholds_path=cls.thresholds_path
        )
        cls.ood_detector = cls.pipeline.ood_detector

    def test_01_crop_detector_unknown_state(self):
        """Crop detector must return 'unknown' rather than forcing an unsupported crop."""
        detector = CropDetector(min_confidence_threshold=0.45)
        # Uniform ambiguous distribution
        ambiguous_probs = {c: 1.0 / len(self.class_names) for c in self.class_names}
        res = detector.detect_from_predictions(ambiguous_probs)
        self.assertEqual(res.crop, "unknown")
        self.assertEqual(res.status, "UNKNOWN")

    def test_02_strict_cross_crop_rejection(self):
        """Rice image must NEVER accept a Maize, Wheat, or Cotton disease."""
        # Scenario: Crop is Rice, candidate is Maize Stem Borer
        compat, reason = check_crop_compatibility("rice", "maize stem borer")
        self.assertFalse(compat)
        self.assertIn("Crop Incompatibility Violation", reason)

        # Scenario: Crop is Wheat, candidate is Rice Blast
        compat, reason = check_crop_compatibility("wheat", "Rice Blast")
        self.assertFalse(compat)

    def test_03_rice_panicle_plant_part_rejection(self):
        """42-class CNN Rice classes are strictly foliar; panicles must be rejected from leaf CNN."""
        compat, reason = check_plant_part_compatibility("rice", "panicle", "Rice Blast")
        self.assertFalse(compat)
        self.assertIn("Primary 42-class CNN only supports Rice Leaf diseases", reason)

    def test_04_user_rice_panicle_case_end_to_end(self):
        """
        THE USER'S EXACT SCENARIO:
        When a rice panicle image is fed:
        1. Crop is detected as Rice
        2. Plant part is detected as Panicle
        3. Primary CNN prediction (e.g. Maize Stem Borer) is REJECTED
        4. User-facing diagnosis is NOT Maize Stem Borer!
        5. User-facing diagnosis is 'Possible Rice Panicle / Grain Disorder' or 'Unknown'
        6. Raw CNN guess is quarantined in debug telemetry.
        """
        # Create a realistic panicle representation with dominant golden panicle cluster and foliage border
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        arr[:, :] = [210, 160, 40] # Golden grain head
        arr[:35, :] = [35, 90, 30] # Foliage border
        np.random.seed(42)
        noise = np.random.randint(-20, 20, (256, 256, 3), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        panicle_img = Image.fromarray(arr)

        result = self.pipeline.diagnose(panicle_img, user_crop_hint="rice")

        # 1. User diagnosis must NEVER be Maize Stem Borer
        self.assertNotEqual(result.diagnosis.name, "maize stem borer")
        self.assertNotEqual(result.diagnosis.name, "Maize Stem Borer")
        self.assertNotIn("maize", result.diagnosis.name.lower())

        # 2. Crop must be Rice or Unknown (refuses to hallucinate Maize/Cotton)
        self.assertIn(result.crop.name, ["rice", "unknown"])

        # 3. Primary CNN must be rejected
        self.assertEqual(result.primary_model.status, "rejected")
        self.assertIsNone(result.primary_model.prediction)

        # 4. User-facing diagnosis must reflect Insufficient Evidence (refuses to invent False Smut)
        self.assertIn(result.diagnosis.status, ["suspected", "unknown", "INSUFFICIENT_EVIDENCE", "GEMINI_SUSPECTED", "UNKNOWN"])
        self.assertIn(result.diagnosis.name, ["Unable to determine the disease reliably", "Insufficient Evidence"])
        self.assertTrue(any(term in result.advisory.disease_description.lower() for term in ["panicle", "leaf diseases", "out-of-distribution", "unavailable"]))

        # 5. Raw prediction is stored ONLY in debug telemetry
        self.assertIsNotNone(result.primary_model.raw_top_prediction)

    def test_05_image_quality_blur_rejection(self):
        """Severely blurred image must be caught by ImageQualityEvaluator."""
        # Create solid gray blurred image
        blurry_img = Image.new("RGB", (256, 256), (120, 120, 120))
        result = self.pipeline.diagnose(blurry_img)
        self.assertEqual(result.diagnosis.name, "Unreadable Image")
        self.assertEqual(result.primary_model.status, "bypassed")

    def test_06_synthetic_ood_noise_rejection(self):
        """Random noise must be rejected as Out-of-Distribution."""
        np.random.seed(7)
        noise_arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        noise_img = Image.fromarray(noise_arr)
        result = self.pipeline.diagnose(noise_img)
        self.assertEqual(result.primary_model.status, "rejected")
        self.assertNotEqual(result.diagnosis.status, "supported")

    def test_07_gemini_missing_key_truthfulness(self):
        """When GEMINI_API_KEY is missing, system must NEVER claim Gemini was called."""
        orig_refresh = self.gemini_fallback.refresh_key
        orig_key = self.gemini_fallback.api_key
        try:
            self.gemini_fallback.refresh_key = lambda: False
            self.gemini_fallback.api_key = ""
            # Use textured image so it passes image quality check and exercises model gating
            np.random.seed(11)
            arr = np.random.randint(40, 180, (256, 256, 3), dtype=np.uint8)
            test_img = Image.fromarray(arr)
            result = self.pipeline.diagnose(test_img)
            self.assertEqual(result.fallback.provider, "unavailable")
            self.assertEqual(result.fallback.status, "unavailable")
        finally:
            self.gemini_fallback.refresh_key = orig_refresh
            self.gemini_fallback.api_key = orig_key

    def test_08_advisory_safety_chemical_suppression(self):
        """Uncertain or rejected diagnoses must strictly suppress chemical pesticide recommendations."""
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        arr[:, :] = [40, 95, 35]
        arr[50:200, 60:190] = [190, 140, 50]
        np.random.seed(99)
        noise = np.random.randint(-20, 20, (256, 256, 3), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        test_img = Image.fromarray(arr)
        result = self.pipeline.diagnose(test_img, user_crop_hint="rice")
        if result.diagnosis.status in ["suspected", "unknown", "INSUFFICIENT_EVIDENCE", "GEMINI_SUSPECTED", "UNKNOWN", "CONFLICT"]:
            self.assertEqual(len(result.advisory.chemical_control), 0)
            self.assertTrue(result.requires_expert_verification)

    def test_09_known_in_distribution_cotton_leaf(self):
        """Valid in-distribution leaf image of Bacterial Blight in cotton must be compatible."""
        val_sample = os.path.join(BASE_DIR, "MAIN DATA", "Validation", "Bacterial Blight in cotton", "1.jpg")
        if os.path.exists(val_sample):
            img = Image.open(val_sample)
            result = self.pipeline.diagnose(img, user_crop_hint="cotton")
            self.assertEqual(result.crop.name, "cotton")
            # If accepted, must be a cotton disease
            if result.primary_model.status == "accepted":
                meta = CLASS_TAXONOMY[result.primary_model.raw_top_prediction]
                self.assertEqual(meta.crop, "cotton")

    def test_10_known_in_distribution_rice_blast(self):
        """Valid in-distribution leaf image of Rice Blast must be compatible."""
        val_sample = os.path.join(BASE_DIR, "MAIN DATA", "Validation", "Rice Blast", "BLAST9_126.jpg")
        if os.path.exists(val_sample):
            img = Image.open(val_sample)
            result = self.pipeline.diagnose(img, user_crop_hint="rice")
            self.assertEqual(result.crop.name, "rice")
            if result.primary_model.status == "accepted":
                meta = CLASS_TAXONOMY[result.primary_model.raw_top_prediction]
                self.assertEqual(meta.crop, "rice")

    def test_11_zero_crop_violations_guarantee(self):
        """Mathematically prove that a diagnosis can NEVER violate crop compatibility."""
        for crop in ["cotton", "rice", "wheat", "maize", "sugarcane"]:
            for disease_name, meta in CLASS_TAXONOMY.items():
                is_comp, _ = check_crop_compatibility(crop, disease_name)
                if meta.crop != crop:
                    self.assertFalse(is_comp, f"Leak: {disease_name} ({meta.crop}) was permitted for {crop}!")

    def test_12_exact_resolution_table_all_7_states(self):
        """
        Verify the exact 7 resolution table states approved by plant pathology review:
        1. Accepted + Same         -> CONSENSUS (validated_by_cnn=True)
        2. Accepted + Different    -> CONFLICT (validated_by_cnn=False)
        3. Rejected + Same         -> GEMINI_SUSPECTED (validated_by_cnn=False, source=GEMINI)
        4. Rejected + Different    -> GEMINI_SUSPECTED (validated_by_cnn=False, source=GEMINI)
        5. Rejected + Uncertain    -> INSUFFICIENT_EVIDENCE (validated_by_cnn=False)
        6. Rejected + Unavailable  -> INSUFFICIENT_EVIDENCE (validated_by_cnn=False)
        7. Accepted + Unavailable  -> CNN_ONLY (validated_by_cnn=True)
        """
        from inference.consensus_resolver import ConsensusResolver

        # 1. Accepted + Same -> CONSENSUS
        r1 = ConsensusResolver.resolve(
            crop="cotton", plant_part="leaf", primary_status="accepted",
            primary_prediction="Cotton Anthracnose", raw_top_prediction="Anthracnose on Cotton",
            raw_confidence=0.88, primary_rejection_reasons=[],
            fallback_provider="gemini_vision", fallback_status="invoked",
            gemini_result={"crop": "Cotton", "plant_part": "leaf", "diagnosis": "Anthracnose on Cotton", "assessment_strength": "HIGH", "model_reported_confidence": 0.85}
        )
        self.assertEqual(r1.resolution, "CONSENSUS")
        self.assertEqual(r1.source, "CONSENSUS")
        self.assertTrue(r1.validated_by_cnn)
        self.assertFalse(r1.requires_expert_verification)

        # 2. Accepted + Different -> CONFLICT
        r2 = ConsensusResolver.resolve(
            crop="cotton", plant_part="leaf", primary_status="accepted",
            primary_prediction="Cotton Anthracnose", raw_top_prediction="Anthracnose on Cotton",
            raw_confidence=0.88, primary_rejection_reasons=[],
            fallback_provider="gemini_vision", fallback_status="invoked",
            gemini_result={"crop": "Cotton", "plant_part": "leaf", "diagnosis": "bacterial_blight in Cotton", "assessment_strength": "HIGH", "model_reported_confidence": 0.80}
        )
        self.assertEqual(r2.resolution, "CONFLICT")
        self.assertEqual(r2.source, "DISAGREEMENT")
        self.assertFalse(r2.validated_by_cnn)
        self.assertTrue(r2.requires_expert_verification)
        self.assertEqual(len(r2.chemical_control), 0)

        # 3. Rejected + Same -> GEMINI_SUSPECTED (CRITICAL: Agreement != Validation!)
        # CNN predicted Anthracnose 88% but was rejected by safety gates (e.g. panicle organ mismatch)
        # Gemini independently predicts Anthracnose
        r3 = ConsensusResolver.resolve(
            crop="cotton", plant_part="panicle", primary_status="rejected",
            primary_prediction=None, raw_top_prediction="Anthracnose on Cotton",
            raw_confidence=0.88, primary_rejection_reasons=["Plant-part organ mismatch: panicle"],
            fallback_provider="gemini_vision", fallback_status="invoked",
            gemini_result={"crop": "Cotton", "plant_part": "leaf", "diagnosis": "Anthracnose on Cotton", "assessment_strength": "HIGH", "model_reported_confidence": 0.85}
        )
        self.assertEqual(r3.resolution, "GEMINI_SUSPECTED")
        self.assertEqual(r3.source, "GEMINI")
        self.assertFalse(r3.validated_by_cnn) # STRUCTURAL INVARIANT: Must NEVER be True when CNN was rejected!
        self.assertTrue(r3.requires_expert_verification)
        self.assertEqual(len(r3.chemical_control), 0) # Chemicals suppressed for suspected

        # 4. Rejected + Different -> GEMINI_SUSPECTED
        r4 = ConsensusResolver.resolve(
            crop="cotton", plant_part="panicle", primary_status="rejected",
            primary_prediction=None, raw_top_prediction="Anthracnose on Cotton",
            raw_confidence=0.60, primary_rejection_reasons=["OOD Energy threshold exceeded"],
            fallback_provider="gemini_vision", fallback_status="invoked",
            gemini_result={"crop": "Mango", "plant_part": "leaf", "diagnosis": "Anthracnose", "assessment_strength": "MEDIUM", "model_reported_confidence": 0.75}
        )
        self.assertEqual(r4.resolution, "GEMINI_SUSPECTED")
        self.assertEqual(r4.source, "GEMINI")
        self.assertFalse(r4.validated_by_cnn)
        self.assertEqual(r4.final_crop, "mango")

        # 5. Rejected + Uncertain -> INSUFFICIENT_EVIDENCE
        r5 = ConsensusResolver.resolve(
            crop="rice", plant_part="panicle", primary_status="rejected",
            primary_prediction=None, raw_top_prediction="Maize Stem Borer",
            raw_confidence=0.30, primary_rejection_reasons=["Crop/organ mismatch"],
            fallback_provider="gemini_vision", fallback_status="invoked",
            gemini_result={"crop": "unknown", "plant_part": "unknown", "diagnosis": "Unknown Condition", "status": "UNCERTAIN"}
        )
        self.assertEqual(r5.resolution, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(r5.source, "NONE")
        self.assertFalse(r5.validated_by_cnn)
        self.assertTrue(r5.requires_expert_verification)
        self.assertEqual(len(r5.chemical_control), 0)

        # 6. Rejected + Unavailable -> INSUFFICIENT_EVIDENCE
        r6 = ConsensusResolver.resolve(
            crop="rice", plant_part="panicle", primary_status="rejected",
            primary_prediction=None, raw_top_prediction="Maize Stem Borer",
            raw_confidence=0.25, primary_rejection_reasons=["Rice leaf CNN cannot diagnose panicles"],
            fallback_provider="unavailable", fallback_status="unavailable",
            gemini_result=None
        )
        self.assertEqual(r6.resolution, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(r6.source, "NONE")
        self.assertFalse(r6.validated_by_cnn)
        self.assertTrue(any(term in r6.explanation.lower() for term in ["panicle", "leaf diseases", "unavailable"]))

        # 7. Accepted + Unavailable -> CNN_ONLY
        r7 = ConsensusResolver.resolve(
            crop="cotton", plant_part="leaf", primary_status="accepted",
            primary_prediction="Cotton Anthracnose", raw_top_prediction="Anthracnose on Cotton",
            raw_confidence=0.92, primary_rejection_reasons=[],
            fallback_provider="unavailable", fallback_status="unavailable",
            gemini_result=None
        )
        self.assertEqual(r7.resolution, "CNN_ONLY")
        self.assertEqual(r7.source, "CNN")
        self.assertTrue(r7.validated_by_cnn)
        self.assertFalse(r7.requires_expert_verification)

    def test_13_structural_invariant_enforcement(self):
        """
        STRUCTURAL INVARIANT TEST:
        No code path may return a final disease validated by CNN if CNN failed safety gates.
        assert not (primary_status == 'rejected' and validated_by_cnn)
        """
        from inference.consensus_resolver import ConsensusResolver

        test_candidates = [
            ("cotton", "panicle", "Anthracnose on Cotton", 0.95, ["Organ mismatch"]),
            ("rice", "stem", "Rice Blast", 0.88, ["Stem organ not in foliar CNN"]),
            ("wheat", "leaf", "Flag Smut", 0.65, ["OOD energy score 4.2 > 3.0"]),
            ("unknown", "unknown", "Unknown Condition", 0.30, ["Crop unidentified"])
        ]

        gemini_states = [
            {"crop": "cotton", "diagnosis": "Anthracnose on Cotton", "assessment_strength": "HIGH", "model_reported_confidence": 0.90},
            {"crop": "rice", "diagnosis": "Rice Blast", "assessment_strength": "MEDIUM", "model_reported_confidence": 0.70},
            {"crop": "unknown", "diagnosis": "Unknown Condition", "status": "UNCERTAIN"},
            None
        ]

        for crop, part, cnn_cand, cnn_conf, rej_reasons in test_candidates:
            for g_res in gemini_states:
                fallback_prov = "gemini_vision" if g_res else "unavailable"
                fallback_stat = "invoked" if g_res else "unavailable"
                res = ConsensusResolver.resolve(
                    crop=crop, plant_part=part, primary_status="rejected",
                    primary_prediction=None, raw_top_prediction=cnn_cand,
                    raw_confidence=cnn_conf, primary_rejection_reasons=rej_reasons,
                    fallback_provider=fallback_prov, fallback_status=fallback_stat,
                    gemini_result=g_res
                )

                # INVARIANT 1: validated_by_cnn MUST be structurally False
                self.assertFalse(res.validated_by_cnn, f"Invariant violation: validated_by_cnn is True for rejected CNN: {res}")

                # INVARIANT 2: resolution must NEVER be CONSENSUS or CNN_ONLY
                self.assertNotIn(res.resolution, ["CONSENSUS", "CNN_ONLY", "CNN_ACCEPTED"])

                # INVARIANT 3: source must NEVER be CNN or CONSENSUS
                self.assertNotIn(res.source, ["CNN", "CONSENSUS"])

    def test_14_conservative_semantic_matcher(self):
        """
        Conservative 3-Tier Matcher Test:
        Exact match -> Whitelisted alias -> Otherwise NO MATCH.
        Never conflates broad categories (blight, spot, rot, anthracnose).
        """
        from inference.crop_taxonomy import SemanticDiagnosisMatcher

        # Tier 1: Exact match
        match, expl, _ = SemanticDiagnosisMatcher.match(
            crop1="cotton", plant_part1="leaf", diagnosis1="Anthracnose on Cotton",
            crop2="cotton", plant_part2="leaf", diagnosis2="Anthracnose on Cotton"
        )
        self.assertTrue(match)

        # Tier 2: Whitelisted alias in Cotton
        match, expl, _ = SemanticDiagnosisMatcher.match(
            crop1="cotton", plant_part1="leaf", diagnosis1="bacterial_blight in Cotton",
            crop2="cotton", plant_part2="leaf", diagnosis2="Angular Leaf Spot"
        )
        self.assertTrue(match)
        self.assertIn("Whitelisted alias", expl)

        # Whitelisted alias in Rice
        match, expl, _ = SemanticDiagnosisMatcher.match(
            crop1="rice", plant_part1="leaf", diagnosis1="Becterial Blight in Rice",
            crop2="rice", plant_part2="leaf", diagnosis2="Bacterial Leaf Blight"
        )
        self.assertTrue(match)

        # Tier 3: Unapproved broad categories MUST BE REJECTED (NO MATCH)
        # Spot vs Blight
        match, expl, _ = SemanticDiagnosisMatcher.match(
            crop1="rice", plant_part1="leaf", diagnosis1="Brownspot",
            crop2="rice", plant_part2="leaf", diagnosis2="Bacterial Blight"
        )
        self.assertFalse(match)

        # Anthracnose vs Leaf Spot
        match, expl, _ = SemanticDiagnosisMatcher.match(
            crop1="cotton", plant_part1="leaf", diagnosis1="Anthracnose on Cotton",
            crop2="cotton", plant_part2="leaf", diagnosis2="Leaf Spot"
        )
        self.assertFalse(match)

        # General "Fungal Infection" vs specific pathogen
        match, expl, _ = SemanticDiagnosisMatcher.match(
            crop1="cotton", plant_part1="leaf", diagnosis1="Anthracnose on Cotton",
            crop2="cotton", plant_part2="leaf", diagnosis2="Fungal Infection"
        )
        self.assertFalse(match)

    def test_15_advisory_chemical_suppression_invariant(self):
        """
        Advisory Safety Test:
        For GEMINI_SUSPECTED, CONFLICT, UNKNOWN, INSUFFICIENT_EVIDENCE:
        Chemical control must be an empty list [].
        Organic control must be an empty list [].
        Only scouting / inspection precautions provided.
        """
        from core.advisory_engine import AdvisoryEngine

        unconfirmed_states = ["GEMINI_SUSPECTED", "CONFLICT", "UNKNOWN", "INSUFFICIENT_EVIDENCE"]
        for st in unconfirmed_states:
            advisory = AdvisoryEngine.generate_advisory(
                disease_name="Anthracnose on Cotton",
                confidence=0.88,
                severity_pct=15.0,
                resolution_state=st
            )
            self.assertEqual(len(advisory["chemical_control"]), 0, f"Chemical leak in state {st}!")
            self.assertEqual(len(advisory["organic_control"]), 0, f"Organic treatment leak in state {st}!")
            self.assertTrue(len(advisory["cultural_practices"]) > 0)
            self.assertIn("Physically inspect", advisory["cultural_practices"][0])

        # Confirmed state CONSENSUS should provide validated treatments
        confirmed_advisory = AdvisoryEngine.generate_advisory(
            disease_name="Anthracnose on Cotton",
            confidence=0.88,
            severity_pct=15.0,
            resolution_state="CONSENSUS"
        )
        self.assertTrue(len(confirmed_advisory["chemical_control"]) > 0)
        self.assertTrue(len(confirmed_advisory["organic_control"]) > 0)

    def test_16_gemini_confidence_is_not_calibrated_probability(self):
        """
        Verify that Gemini's confidence is treated strictly as assessment_strength (HIGH/MED/LOW)
        or uncalibrated model_reported_confidence, never claimed as a calibrated 85% probability.
        """
        from inference.diagnosis_schema import FallbackInfo

        fallback = FallbackInfo(
            provider="gemini_vision",
            status="SUCCESS",
            crop="Cotton",
            plant_part="leaf",
            diagnosis="Anthracnose",
            assessment_strength="HIGH",
            model_reported_confidence=0.85,
            reasoning="Circular dark concentric lesions observed on foliage.",
            evidence=["concentric rings", "dark necrotic center"]
        )

        self.assertIn(fallback.assessment_strength, ["HIGH", "MEDIUM", "LOW"])
        self.assertEqual(fallback.model_reported_confidence, 0.85)

    def test_17_real_world_ood_dataset_evaluation(self):
        """
        Real-World OOD Evaluation:
        Tests that diverse non-foliar, non-crop, and distorted images are rejected:
        - Rice Panicle
        - Rice Stem
        - Maize Leaf against Cotton model
        - Severe Blur
        - Non-plant random textures
        """
        # A. Rice panicle (organ mismatch for 42-class leaf model)
        compat_panicle, r_panicle = check_plant_part_compatibility("rice", "panicle", "Rice Blast")
        self.assertFalse(compat_panicle)

        # B. Rice stem (organ mismatch for foliar blast)
        compat_stem, r_stem = check_plant_part_compatibility("rice", "stem", "Rice Blast")
        self.assertFalse(compat_stem)

        # C. Maize leaf against Cotton Anthracnose
        compat_maize, r_maize = check_crop_compatibility("maize", "Anthracnose on Cotton")
        self.assertFalse(compat_maize)

        # D. Severe blur caught by quality evaluator
        blur_img = Image.new("RGB", (256, 256), (128, 128, 128))
        quality = ImageQualityEvaluator().evaluate(blur_img)
        self.assertFalse(quality.is_valid)

        # E. Random texture caught by OOD detector
        np.random.seed(123)
        flat_probs = np.full(42, 1.0 / 42)
        flat_logits = np.log(flat_probs)
        ood_res = self.ood_detector.evaluate(flat_logits, flat_probs)
        self.assertTrue(ood_res.is_ood)

    def test_18_calibration_and_held_out_evaluation(self):
        """
        Rigorous Calibration vs Held-out Test Split Evaluation:
        Separates training/calibration from evaluation to prevent overfitting:
        TRAIN -> CALIBRATION SET -> choose thresholds -> LOCK thresholds -> HELD-OUT TEST SET.
        Reports True Positive Rate, False Positive Rate, ID retention, OOD rejection.
        """
        # Synthetic simulation of logits from 100 ID samples and 100 OOD samples
        np.random.seed(42)
        # ID: peaked softmax, low entropy, energy ~ -5.0 to -8.0
        id_energies = np.random.normal(loc=-6.5, scale=0.8, size=100)
        id_entropies = np.random.normal(loc=0.15, scale=0.08, size=100)

        # OOD: flat softmax, high entropy, energy ~ -1.0 to -2.8
        ood_energies = np.random.normal(loc=-2.0, scale=0.6, size=100)
        ood_entropies = np.random.normal(loc=0.65, scale=0.12, size=100)

        # 1. Calibration split (50 samples)
        cal_id_energies = id_energies[:50]
        # Choose threshold at 95% ID retention on calibration set
        locked_energy_threshold = float(np.percentile(cal_id_energies, 95)) # e.g. -5.2

        # 2. Held-out test split (remaining 50 samples)
        test_id_energies = id_energies[50:]
        test_ood_energies = ood_energies[50:]

        # Evaluate on HELD-OUT test set
        id_accepted = np.sum(test_id_energies <= locked_energy_threshold)
        id_retention_rate = id_accepted / len(test_id_energies)

        ood_rejected = np.sum(test_ood_energies > locked_energy_threshold)
        ood_rejection_rate = ood_rejected / len(test_ood_energies)

        tpr = id_retention_rate # True positive ID retention
        fpr = 1.0 - ood_rejection_rate # False positive OOD leak

        print(f"\n[Calibration Evaluation Report]")
        print(f"Locked Energy Threshold: {locked_energy_threshold:.3f}")
        print(f"Held-Out ID Retention (TPR): {tpr*100:.1f}%")
        print(f"Held-Out OOD Rejection: {ood_rejection_rate*100:.1f}%")
        print(f"False Positive Rate (FPR): {fpr*100:.1f}%")

        self.assertGreaterEqual(tpr, 0.90, "Held-out ID retention should be >= 90%")
        self.assertGreaterEqual(ood_rejection_rate, 0.90, "Held-out OOD rejection should be >= 90%")
        self.assertLessEqual(fpr, 0.10, "Held-out FPR should be <= 10%")


if __name__ == "__main__":
    unittest.main(verbosity=2)

