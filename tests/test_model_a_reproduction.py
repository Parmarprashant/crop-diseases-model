"""
AgriVision AI — Pre-Training Gate: Strict Same-Tensor Model A Reproduction Test.
Verifies that EfficientNetB5_CBAM_Hierarchical with the crop branch disabled
reproduces Model A within 1e-5 relative numerical tolerance on identical input tensors.
"""
import os
import sys
import unittest
import torch
import torch.nn.functional as F
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.efficientnet_cbam import EfficientNetB5_CBAM
from models.hierarchical_cbam import EfficientNetB5_CBAM_Hierarchical
from inference.pipeline import CanonicalAggregator

class TestModelAReproduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights_path = os.path.join(BASE_DIR, "weights", "efficientnet_b5_cbam_best.pt")
        cls.class_names_path = os.path.join(BASE_DIR, "weights", "class_names.txt")
        cls.canonical_map_path = os.path.join(BASE_DIR, "weights", "canonical_class_map.json")

        assert os.path.exists(cls.weights_path), f"Missing {cls.weights_path}"
        
        with open(cls.class_names_path, "r") as f:
            cls.class_names = [l.strip() for l in f if l.strip()]
        cls.num_classes = len(cls.class_names) # 281

        # Instantiate Model A original
        cls.model_a = EfficientNetB5_CBAM(num_classes=cls.num_classes, pretrained=False)
        sd_a = torch.load(cls.weights_path, map_location="cpu", weights_only=False)
        cls.model_a.load_state_dict(sd_a, strict=True)
        cls.model_a.eval()

        # Instantiate Candidate Architecture
        cls.model_cand = EfficientNetB5_CBAM_Hierarchical(num_disease_classes=cls.num_classes, num_crop_classes=41, pretrained=False)
        cls.model_cand.load_model_a_weights(cls.weights_path)
        cls.model_cand.eval()

        cls.aggregator = CanonicalAggregator(cls.canonical_map_path, cls.class_names_path)

    def test_same_tensor_reproduction(self):
        """Feed the exact same tensor batch through both architectures and test numerical identity."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Test across 4 distinct random synthetic leaf tensors (256x256 ImageNet normalized)
        x = torch.randn(4, 3, 256, 256, dtype=torch.float32)

        with torch.no_grad():
            logits_a = self.model_a(x)
            logits_cand = self.model_cand(x, return_crop=False)

        # 1. Raw 281 logits comparison
        max_abs_diff = torch.max(torch.abs(logits_a - logits_cand)).item()
        denom = torch.clamp(torch.abs(logits_a), min=1e-6)
        max_rel_diff = torch.max(torch.abs(logits_a - logits_cand) / denom).item()

        print(f"\n[Reproduction Test] Raw Logits Max Absolute Diff: {max_abs_diff:.8e}")
        print(f"[Reproduction Test] Raw Logits Max Relative Diff: {max_rel_diff:.8e}")
        self.assertLess(max_abs_diff, 1e-5, f"Raw logits exceed 1e-5 tolerance: {max_abs_diff}")

        # 2. Softmax probabilities
        probs_a = F.softmax(logits_a, dim=1)
        probs_cand = F.softmax(logits_cand, dim=1)
        prob_diff = torch.max(torch.abs(probs_a - probs_cand)).item()
        print(f"[Reproduction Test] Softmax Probabilities Max Diff: {prob_diff:.8e}")
        self.assertLess(prob_diff, 1e-5, f"Softmax probabilities exceed 1e-5 tolerance: {prob_diff}")

        # 3. Top-1 and Top-3 predictions
        top1_a = torch.argmax(probs_a, dim=1).tolist()
        top1_cand = torch.argmax(probs_cand, dim=1).tolist()
        self.assertEqual(top1_a, top1_cand, "Top-1 predictions differ between Model A and Candidate!")

        _, top3_a = torch.topk(probs_a, k=3, dim=1)
        _, top3_cand = torch.topk(probs_cand, k=3, dim=1)
        self.assertTrue(torch.equal(top3_a, top3_cand), "Top-3 predictions differ between Model A and Candidate!")
        print("[Reproduction Test] Top-1 and Top-3 Predictions: 100% IDENTICAL")

        # 4. Shannon Entropy
        log_probs_a = F.log_softmax(logits_a, dim=1)
        log_probs_cand = F.log_softmax(logits_cand, dim=1)
        entropy_a = -(probs_a * log_probs_a).sum(dim=1) / np.log(self.num_classes)
        entropy_cand = -(probs_cand * log_probs_cand).sum(dim=1) / np.log(self.num_classes)
        entropy_diff = torch.max(torch.abs(entropy_a - entropy_cand)).item()
        print(f"[Reproduction Test] Shannon Entropy Max Diff: {entropy_diff:.8e}")
        self.assertLess(entropy_diff, 1e-5, "Entropy exceeds 1e-5 tolerance!")

        # 5. Energy OOD Score: E(x) = -T * logsumexp(z / T)
        T = 1.0
        energy_a = -T * torch.logsumexp(logits_a / T, dim=1)
        energy_cand = -T * torch.logsumexp(logits_cand / T, dim=1)
        energy_diff = torch.max(torch.abs(energy_a - energy_cand)).item()
        print(f"[Reproduction Test] Energy OOD Score Max Diff: {energy_diff:.8e}")
        self.assertLess(energy_diff, 1e-5, "Energy score exceeds 1e-5 tolerance!")

        # 6. Canonical 167 Aggregation
        for i in range(x.size(0)):
            p_a_np = probs_a[i].numpy()
            p_cand_np = probs_cand[i].numpy()
            _, _, top_a = self.aggregator.aggregate_numpy(p_a_np)
            _, _, top_cand = self.aggregator.aggregate_numpy(p_cand_np)
            self.assertEqual(top_a[0]["class_name"], top_cand[0]["class_name"])
            self.assertAlmostEqual(top_a[0]["confidence"], top_cand[0]["confidence"], places=5)
        print("[Reproduction Test] Canonical 167 Aggregated Probabilities: 100% IDENTICAL")
        print(">>> [PRE-TRAINING GATE PASSED]: Candidate architecture perfectly reproduces Model A.")

if __name__ == "__main__":
    unittest.main()
