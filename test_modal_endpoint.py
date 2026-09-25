import urllib.request
import json
import io
from PIL import Image

# Create a small dummy green leaf image
img = Image.new('RGB', (256, 256), color=(34, 139, 34))
buf = io.BytesIO()
img.save(buf, format='JPEG')
buf.seek(0)
img_bytes = buf.read()

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
    f'Content-Type: image/jpeg\r\n\r\n'
).encode('utf-8') + img_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')

req = urllib.request.Request(
    'https://parmarprashant--agrivision-diagnostic-engine-fastapi-app.modal.run/api/v1/diagnose',
    data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)

try:
    with urllib.request.urlopen(req, timeout=30) as res:
        print('STATUS:', res.status)
        resp_data = json.loads(res.read().decode())
        print('RESPONSE:', json.dumps(resp_data, indent=2))
except urllib.error.HTTPError as e:
    print('HTTP ERROR:', e.code, e.read().decode())
except Exception as e:
    print('ERROR:', e)
