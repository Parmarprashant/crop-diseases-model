import urllib.request
import json

img_path = r"Cotton_Leaves\Test\Aphids edited\1.jpg"
with open(img_path, "rb") as f:
    img_bytes = f.read()

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="1.jpg"\r\n'
    f'Content-Type: image/jpeg\r\n\r\n'
).encode('utf-8') + img_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')

req = urllib.request.Request(
    'https://parmarprashant--agrivision-diagnostic-engine-fastapi-app.modal.run/api/v1/diagnose',
    data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)

with urllib.request.urlopen(req, timeout=30) as res:
    d = json.loads(res.read().decode())
    print("KEYS:", list(d.keys()))
    print("CROP:", json.dumps(d.get("crop"), indent=2))
    print("PLANT PART:", json.dumps(d.get("plant_part"), indent=2))
    print("PRIMARY MODEL:", json.dumps(d.get("primary_model"), indent=2))
    print("DIAGNOSIS:", json.dumps(d.get("diagnosis"), indent=2))
    print("CROP OOD TELEMETRY:", json.dumps(d.get("crop_ood_telemetry"), indent=2))
