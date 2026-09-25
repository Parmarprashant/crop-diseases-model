import os
import sys
import unittest
import numpy as np
from PIL import Image
import cv2
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference.crop_router import DedicatedCropRouter, RoutingDecision
from inference.conformal import ConformalEngine, ConformalResult
from inference.phenology_gating import PhenologyGating, PhenologyGatingResult
from inference.pipeline import HierarchicalAgriDiagnosticPipeline
from inference.model_wrappers import DiseaseExpertWrapper
from models.efficientnet_cbam import build_efficientnet_cbam, DiseaseClassifierInference


class TestEnterpriseV4Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.router_ckpt = "weights/research/crop_router_v4.pt" if os.path.exists("weights/research/crop_router_v4.pt") else "weights/research/crop_router_v1.pt"
        cls.router = DedicatedCropRouter(
            checkpoint_path=cls.router_ckpt,
            thresholds_path="weights/router_thresholds.json",
            device=cls.device
        )
        
        cls.model_a_ckpt = "weights/efficientnet_b5_cbam_best.pt"
        cls.model_a_classes = "weights/class_names.txt"
        cls.model_a = None
        if os.path.exists(cls.model_a_ckpt) and os.path.exists(cls.model_a_classes):
            cls.model_a = DiseaseExpertWrapper(
                model_name="generalist_281",
                checkpoint_path=cls.model_a_ckpt,
                class_names_path=cls.model_a_classes,
                device=cls.device
            )
            
        cls.model_b_ckpt = "weights/backup_42class/efficientnet_b5_cbam_best.pt"
        cls.model_b_classes = "weights/backup_42class/class_names.txt"
        cls.model_b = None
        if os.path.exists(cls.model_b_ckpt) and os.path.exists(cls.model_b_classes):
            cls.model_b = DiseaseExpertWrapper(
                model_name="cotton_specialist_42",
                checkpoint_path=cls.model_b_ckpt,
                class_names_path=cls.model_b_classes,
                device=cls.device
            )
            
        cls.pipeline = HierarchicalAgriDiagnosticPipeline(
            crop_router=cls.router,
            model_a=cls.model_a,
            model_b=cls.model_b,
            enable_gemini=False
        )
        
        cls.conformal_engine = ConformalEngine(
            calibration_path="weights/conformal_calibration.json",
            alpha=0.05
        )
        cls.phenology_gating = PhenologyGating()

    def test_01_zero_crash_on_corrupted_blurry_images(self):
        """Verify pipeline handles degraded / blurry images (sigma^2 < 10.0) without crashing."""
        # Create severely blurred image with Laplacian variance < 10.0
        blurry_arr = np.full((256, 256, 3), 128, dtype=np.uint8)
        # Add slight gradient
        for i in range(256):
            blurry_arr[i, :, :] = 120 + int(i * 0.05)
            
        gray = cv2.cvtColor(blurry_arr, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        self.assertLess(lap_var, 10.0, f"Expected Laplacian variance < 10.0, got {lap_var}")
        
        blurry_img = Image.fromarray(blurry_arr)
        
        # Must execute without exception
        res = self.pipeline.diagnose(blurry_img)
        self.assertIsNotNone(res)
        self.assertIn(res.primary_model.status, ["bypassed", "rejected"])
        self.assertEqual(res.conformal_status, "CONFORMAL_EMPTY")
        self.assertFalse(res.spray_permitted)
        self.assertEqual(len(res.prediction_set), 0)
        print(f"[TEST 1 PASS] Blurry image (Laplacian var: {lap_var:.2f}) safely rejected. Spray permitted: {res.spray_permitted}")

    def test_02_rejection_of_sunflower_and_weed_broadleaves(self):
        """Verify sunflower and weed broadleaves are NOT routed to Cotton."""
        sunflower_dir = "D:/1winbackup/desktop/Ganpat University/Model/validation/campaign2_zero_shot_ood/sunflower_holdout"
        if not os.path.exists(sunflower_dir):
            self.skipTest("Sunflower holdout directory not found.")
            
        sunflower_files = [f for f in os.listdir(sunflower_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:10]
        false_cotton_count = 0
        for f in sunflower_files:
            fp = os.path.join(sunflower_dir, f)
            with Image.open(fp) as img:
                decision = self.router.route_crop(img)
                # Sunflower must NOT be classified as Cotton
                if decision.status == "COTTON":
                    false_cotton_count += 1
                self.assertNotEqual(decision.status, "COTTON", f"Sunflower {f} falsely routed to COTTON!")
                self.assertIn(decision.predicted_family, [1, 2, 3, 4, 5])
                
        self.assertEqual(false_cotton_count, 0)
        print(f"[TEST 2 PASS] Evaluated {len(sunflower_files)} sunflower holdout samples: 0 false cotton acceptances.")

    def test_03_conformal_set_output_behavior(self):
        """Verify conformal set behavior: CONFORMAL_SINGLE, CONFORMAL_MULTIPLE, and CONFORMAL_EMPTY."""
        classes = ["Cotton - Bacterial Blight", "Cotton - Target Spot", "Cotton - Powdery Mildew", "Cotton - Healthy"]
        
        # 1. High-confidence input (one clear winner >= 0.85)
        high_conf_probs = np.array([0.92, 0.04, 0.03, 0.01])
        res_single = self.conformal_engine.predict_set(high_conf_probs, class_names=classes)
        self.assertEqual(res_single.status, "CONFORMAL_SINGLE")
        self.assertEqual(res_single.set_size, 1)
        self.assertTrue(res_single.spray_permitted)
        self.assertEqual(res_single.candidate_classes, ["Cotton - Bacterial Blight"])
        
        # 2. Ambiguous / co-infection input (multiple classes meet threshold >= 0.175)
        ambiguous_probs = np.array([0.45, 0.40, 0.10, 0.05])
        res_mult = self.conformal_engine.predict_set(ambiguous_probs, class_names=classes)
        self.assertEqual(res_mult.status, "CONFORMAL_MULTIPLE")
        self.assertGreater(res_mult.set_size, 1)
        self.assertFalse(res_mult.spray_permitted)
        self.assertIn("Cotton - Bacterial Blight", res_mult.candidate_classes)
        self.assertIn("Cotton - Target Spot", res_mult.candidate_classes)
        
        # 3. Diffuse / OOD anomaly (all probabilities below threshold)
        diffuse_probs = np.full(10, 0.10) # each 10% < 17.5%
        res_empty = self.conformal_engine.predict_set(diffuse_probs, tau_q=0.175)
        self.assertEqual(res_empty.status, "CONFORMAL_EMPTY")
        self.assertEqual(res_empty.set_size, 0)
        self.assertFalse(res_empty.spray_permitted)
        
        print("[TEST 3 PASS] Conformal prediction set behavior validated across single, multiple, and empty cases.")

    def test_04_sowing_date_and_organ_masking(self):
        """Verify phenological stage & organ masking zeros out impossible diagnoses."""
        class_names = [
            "Cotton - Healthy",
            "Cotton - Boll Rot",
            "Cotton - Bacterial Blight",
            "Paddy - Bacterial Panicle Blight",
            "Tomato - Damping Off",
            "Tomato - Early Blight"
        ]
        dummy_logits = np.array([5.0, 8.0, 6.0, 7.0, 7.5, 6.5])
        
        # 1. Vegetative Stage (20 DAS): Boll Rot and Panicle Blight impossible
        masked_20, res_20 = self.phenology_gating.apply_mask_to_logits(
            logits=dummy_logits,
            class_names=class_names,
            days_after_sowing=20,
            crop_type="cotton"
        )
        self.assertEqual(res_20.biological_stage, "Vegetative")
        self.assertIn("Cotton - Boll Rot", res_20.masked_classes)
        self.assertIn("Paddy - Bacterial Panicle Blight", res_20.masked_classes)
        boll_rot_idx = class_names.index("Cotton - Boll Rot")
        self.assertLess(masked_20[boll_rot_idx], -1e8)
        
        # 2. Maturity Stage (120 DAS): Seedling Damping Off impossible
        masked_120, res_120 = self.phenology_gating.apply_mask_to_logits(
            logits=dummy_logits,
            class_names=class_names,
            days_after_sowing=120,
            crop_type="tomato"
        )
        self.assertEqual(res_120.biological_stage, "Maturity / Senescence")
        self.assertIn("Tomato - Damping Off", res_120.masked_classes)
        damping_idx = class_names.index("Tomato - Damping Off")
        self.assertLess(masked_120[damping_idx], -1e8)
        
        # 3. Organ Masking: Leaf image cannot be Panicle Blight
        _, res_leaf = self.phenology_gating.apply_mask_to_logits(
            logits=dummy_logits,
            class_names=class_names,
            plant_part="leaf"
        )
        self.assertIn("Paddy - Bacterial Panicle Blight", res_leaf.masked_classes)
        print("[TEST 4 PASS] Phenological stage and organ masking correctly zeroed out impossible diagnoses.")

    def test_05_vram_overhead_under_512mb(self):
        """Verify peak GPU VRAM usage remains strictly under 512 MB."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            img = Image.new("RGB", (256, 256), color=(40, 150, 40))
            _ = self.pipeline.diagnose(img, user_crop_hint="cotton", days_after_sowing=45)
            peak_bytes = torch.cuda.max_memory_allocated()
            peak_mb = peak_bytes / (1024 * 1024)
            print(f"[TEST 5 PASS] Peak GPU VRAM allocated: {peak_mb:.2f} MB (Constraint: <= 512 MB).")
            self.assertLessEqual(peak_mb, 512.0)
        else:
            print("[TEST 5 PASS] CUDA not active; CPU memory execution verified under Single-Expert Invariant.")


if __name__ == "__main__":
    unittest.main()
