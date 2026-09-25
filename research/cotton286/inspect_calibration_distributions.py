import os
import sys
import json
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference.crop_router import CropRouterModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CropRouterModel(pretrained=False)
model.load_state_dict(torch.load("weights/research/crop_router_v1.pt", map_location="cpu", weights_only=False))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Cotton dev
with open("research/cotton286/cotton_split_manifest.json", "r", encoding="utf-8") as f:
    cotton_dev = json.load(f)["splits"]["dev"]

def eval_dir(name, paths):
    p_cottons = []
    energies = []
    with torch.no_grad():
        for p in paths:
            try:
                with Image.open(p) as img:
                    t = transform(img.convert("RGB")).unsqueeze(0).to(device)
                logits = model(t)
                probs = F.softmax(logits, dim=1).squeeze(0)
                energy = -float(torch.logsumexp(logits, dim=1).item())
                p_cottons.append(float(probs[1].item()))
                energies.append(energy)
            except Exception as e:
                pass
    avg_c = sum(p_cottons) / max(1, len(p_cottons))
    avg_e = sum(energies) / max(1, len(energies))
    print(f"[{name}] N={len(p_cottons)} | Avg P(Cotton): {avg_c:.4f} | Avg Energy: {avg_e:.4f}")
    if p_cottons:
        print(f"   Min P(C): {min(p_cottons):.4f}, Max P(C): {max(p_cottons):.4f}")
        print(f"   Min Energy: {min(energies):.4f}, Max Energy: {max(energies):.4f}")

eval_dir("Cotton Dev (15)", [x["file_path"] for x in cotton_dev[:15]])

field_dir = "D:/1winbackup/desktop/Ganpat University/Model/validation/field_test_cohort"
field_files = [os.path.join(field_dir, f) for f in sorted(os.listdir(field_dir))[30:55]]
eval_dir("Field Non-Cotton (25)", field_files)

rose_dir = "D:/1winbackup/desktop/Ganpat University/Model/validation/campaign2_zero_shot_ood/rose_leaf_holdout"
rose_files = [os.path.join(rose_dir, f) for f in os.listdir(rose_dir)[:25]]
eval_dir("Rose Non-Cotton (25)", rose_files)

ood_dir = "D:/1winbackup/desktop/Ganpat University/Model/validation/open_set_v2_cohort"
ood_files = [os.path.join(ood_dir, f) for f in os.listdir(ood_dir)[:25]]
eval_dir("OOD Stress (25)", ood_files)
