import os
import sys
import json
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference.crop_router import DedicatedCropRouter

router = DedicatedCropRouter(
    checkpoint_path="weights/research/crop_router_v1.pt",
    thresholds_path="weights/router_thresholds.json"
)

with open("research/cotton286/cotton_split_manifest.json", "r", encoding="utf-8") as f:
    dev = json.load(f)["splits"]["dev"]

print(f"Total dev samples: {len(dev)}")
for i, item in enumerate(dev[:10]):
    with Image.open(item["file_path"]) as img:
        d = router.route(img)
        print(f"Dev[{i}]: status={d.status}, p_cotton={d.p_cotton:.3f}, margin={d.margin:.3f}, energy={d.ood_score:.3f}, is_ood={d.is_ood}, path={os.path.basename(item['file_path'])}")
