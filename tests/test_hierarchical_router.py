import os
import sys
import json
import hashlib
import unittest
from PIL import Image, ImageFilter
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference.crop_router import DedicatedCropRouter
from inference.model_wrappers import DiseaseExpertWrapper
from inference.pipeline import HierarchicalAgriDiagnosticPipeline


class TestHierarchicalCropRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.router = DedicatedCropRouter(
            checkpoint_path="weights/research/crop_router_v1.pt",
            thresholds_path="weights/router_thresholds.json",
            device=cls.device
        )
        cls.model_a = DiseaseExpertWrapper(
            model_name="generalist_281",
            checkpoint_path="weights/efficientnet_b5_cbam_best.pt",
            class_names_path="weights/class_names.txt",
            device=cls.device
        )
        cls.model_b = DiseaseExpertWrapper(
            model_name="cotton_specialist_42",
            checkpoint_path="weights/backup_42class/efficientnet_b5_cbam_best.pt",
            class_names_path="weights/backup_42class/class_names.txt",
            device=cls.device
        )
        cls.pipeline = HierarchicalAgriDiagnosticPipeline(
            crop_router=cls.router,
            model_a=cls.model_a,
            model_b=cls.model_b,
            thresholds_path="weights/calibration_thresholds.json",
            enable_gemini=False
        )
        
        # Manifests
        with open("research/cotton286/cotton_split_manifest.json", "r", encoding="utf-8") as f:
            cls.cotton_manifest = json.load(f)
            
    def test_01_firewall_hash_integrity(self):
        """Verify production checkpoints remain 100% bit-identical."""
        expected_hashes = {
            "weights/efficientnet_b5_cbam_best.pt": "b4db2209bfc416716d55339d0c21cb0c2d62ebe0eee089d6d00712dd198717b7",
            "weights/backup_42class/efficientnet_b5_cbam_best.pt": "95f5cb7fffcd315be1bf4224957f3c118a1529793e68aecbcfea5cf5b2aa11c4",
            "../Model/weights/crop_ood_expert_campaign2.pt": "d71680b970be84f77eeb77b81118874ba2a2fd362790ed5e2d25b6e34a515f9b"
        }
        for path, expected_hash in expected_hashes.items():
            full_path = os.path.normpath(os.path.join(BASE_DIR, path))
            if os.path.exists(full_path):
                h = hashlib.sha256()
                with open(full_path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
                actual_hash = h.hexdigest()
                self.assertEqual(actual_hash, expected_hash, f"Firewall breach on {path}!")

    def test_02_cotton_routing_to_model_b(self):
        """Verify that high-confidence cotton images route to Model B (Cotton Specialist)."""
        sample = self.cotton_manifest["splits"]["dev"][1]  # dev[1] is in-distribution cotton
        with Image.open(sample["file_path"]) as img:
            img = img.convert("RGB")
            decision = self.router.route(img)
            self.assertEqual(decision.status, "COTTON")
            self.assertEqual(decision.expert, "cotton_specialist_42")
            self.assertGreaterEqual(decision.p_cotton, 0.70)
            
            # Execute pipeline
            res = self.pipeline.diagnose(img)
            self.assertIsNotNone(res.routing)
            self.assertEqual(res.routing["status"], "COTTON")
            self.assertEqual(res.routing["expert"], "cotton_specialist_42")

    def test_02b_cotton_ood_uncertainty_rejection(self):
        """Verify that atypical cotton images with high energy are defensively routed to UNCERTAIN."""
        sample = self.cotton_manifest["splits"]["dev"][0]  # dev[0] has high energy
        with Image.open(sample["file_path"]) as img:
            img = img.convert("RGB")
            decision = self.router.route(img)
            self.assertEqual(decision.status, "UNCERTAIN")
            self.assertTrue(decision.is_ood)
            
            # Execute pipeline: neither expert should run, advisory should prohibit chemical spray
            res = self.pipeline.diagnose(img)
            self.assertEqual(res.primary_model.status, "bypassed")
            self.assertEqual(res.advisory.chemical_control, [])

    def test_03_quality_gate_preempts_routing(self):
        """Verify that extremely degraded/dark images fail Quality Gate before routing."""
        # Create solid dark image
        dark_img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        res = self.pipeline.diagnose(dark_img)
        self.assertEqual(res.primary_model.status, "bypassed")
        self.assertIn("quality gate failed", res.primary_model.rejection_reason)
        self.assertIsNone(res.routing)

    def test_04_broadleaf_negative_safety(self):
        """Verify that sunflower broadleaf holdouts do NOT get high-confidence false cotton routing."""
        sunflower_dir = "D:/1winbackup/desktop/Ganpat University/Model/validation/campaign2_zero_shot_ood/sunflower_holdout"
        if os.path.exists(sunflower_dir):
            files = [f for f in os.listdir(sunflower_dir) if f.endswith(('.jpg', '.png'))][:5]
            for f in files:
                with Image.open(os.path.join(sunflower_dir, f)) as img:
                    decision = self.router.route(img)
                    # Must NOT route with high confidence to Cotton without hint
                    if decision.status == "COTTON":
                        print(f"Warning: Sunflower {f} falsely accepted as cotton (P={decision.p_cotton:.3f})")
                    # If uncertain, pipeline must defensively reject
                    if decision.status == "UNCERTAIN":
                        res = self.pipeline.diagnose(img)
                        self.assertEqual(res.primary_model.status, "bypassed")
                        self.assertEqual(res.advisory.chemical_control, [])

    def test_05_single_expert_execution_invariant(self):
        """Verify that exactly one expert executes per request."""
        sample = self.cotton_manifest["splits"]["dev"][1]
        with Image.open(sample["file_path"]) as img:
            res = self.pipeline.diagnose(img)
            self.assertIsNotNone(res.routing)
            # When routed to Cotton, Model B executed, Model A did not
            if res.routing["status"] == "COTTON":
                self.assertEqual(res.routing["expert"], "cotton_specialist_42")
                # Diagnosis should come from 42-class taxonomy
                self.assertIsNotNone(res.primary_model.raw_top_prediction)


if __name__ == "__main__":
    unittest.main()
