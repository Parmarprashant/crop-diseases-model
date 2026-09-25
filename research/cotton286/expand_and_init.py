import os
import sys
import json
import hashlib
import torch
import torch.nn as nn

def expand_and_initialize():
    orig_path = "weights/efficientnet_b5_cbam_best.pt"
    expected_hash = "b4db2209bfc416716d55339d0c21cb0c2d62ebe0eee089d6d00712dd198717b7"
    
    # Check original checkpoint hash
    if not os.path.exists(orig_path):
        print(f"Error: {orig_path} not found.")
        sys.exit(1)
        
    with open(orig_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
        
    if actual_hash != expected_hash:
        print(f"FIREWALL ERROR: Checkpoint hash mismatch!")
        print(f"Expected: {expected_hash}")
        print(f"Actual:   {actual_hash}")
        sys.exit(1)
        
    print(f"Production Model A Firewall Verified: {actual_hash}")
    
    # Load original checkpoint
    orig_sd = torch.load(orig_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in orig_sd:
        sd = orig_sd["model_state_dict"]
    else:
        sd = orig_sd
        
    old_weight = sd["classifier.4.weight"]  # [281, 512]
    old_bias = sd["classifier.4.bias"]      # [281]
    
    print(f"Original classifier.4.weight shape: {old_weight.shape}")
    print(f"Original classifier.4.bias shape:   {old_bias.shape}")
    
    # Deterministic seed for new row initialization
    torch.manual_seed(42)
    
    new_weight = torch.empty(286, 512, dtype=old_weight.dtype)
    new_bias = torch.zeros(286, dtype=old_bias.dtype)
    
    # Copy bit-for-bit indices 0..280
    new_weight[:281].copy_(old_weight)
    new_bias[:281].copy_(old_bias)
    
    # Initialize new rows 281..285 with Kaiming Normal
    nn.init.kaiming_normal_(new_weight[281:286], mode="fan_out", nonlinearity="relu")
    nn.init.zeros_(new_bias[281:286])
    
    # Verification checks
    weight_identical = torch.equal(new_weight[:281], old_weight)
    bias_identical = torch.equal(new_bias[:281], old_bias)
    new_rows_nonzero = bool(torch.all(new_weight[281:286] != 0).item())
    
    print(f"Bit-for-bit check (0..280): Weight={weight_identical}, Bias={bias_identical}")
    print(f"New rows (281..285) properly initialized: {new_rows_nonzero}")
    
    if not (weight_identical and bias_identical and new_rows_nonzero):
        print("Error: Initialization validation failed!")
        sys.exit(1)
        
    # Build new state dict
    new_sd = dict(sd)
    new_sd["classifier.4.weight"] = new_weight
    new_sd["classifier.4.bias"] = new_bias
    
    # Save initialization checkpoint
    init_path = os.path.join("research", "cotton286", "init_286.pt")
    torch.save(new_sd, init_path)
    
    with open(init_path, "rb") as f:
        init_hash = hashlib.sha256(f.read()).hexdigest()
        
    class_mapping = {
        281: "Cotton - Aphids",
        282: "Cotton - Bacterial Blight",
        283: "Cotton - Healthy",
        284: "Cotton - Powdery Mildew",
        285: "Cotton - Target Spot"
    }
    
    audit_report = {
        "status": "EXPANSION_INITIALIZATION_VERIFIED",
        "original_checkpoint": {
            "path": orig_path,
            "sha256": actual_hash,
            "classes": 281,
            "weight_shape": list(old_weight.shape),
            "bias_shape": list(old_bias.shape)
        },
        "expanded_checkpoint": {
            "path": init_path,
            "sha256": init_hash,
            "classes": 286,
            "weight_shape": list(new_weight.shape),
            "bias_shape": list(new_bias.shape)
        },
        "new_class_mapping": class_mapping,
        "verification": {
            "weight_rows_0_to_280_identical": weight_identical,
            "bias_rows_0_to_280_identical": bias_identical,
            "new_rows_281_to_285_initialized": new_rows_nonzero,
            "initialization_method": "kaiming_normal_fan_out_relu",
            "seed": 42
        },
        "row_statistics": {
            "old_rows_weight_mean": float(new_weight[:281].mean().item()),
            "old_rows_weight_std": float(new_weight[:281].std().item()),
            "new_rows_weight_mean": float(new_weight[281:286].mean().item()),
            "new_rows_weight_std": float(new_weight[281:286].std().item())
        }
    }
    
    audit_out = os.path.join("research", "cotton286", "class_expansion_audit.json")
    with open(audit_out, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
        
    print(f"Saved init checkpoint: {init_path} (SHA256: {init_hash[:16]}...)")
    print(f"Saved audit report: {audit_out}")

if __name__ == "__main__":
    expand_and_initialize()
