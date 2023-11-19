import requests
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from io import BytesIO
#digits dataset
digits = load_digits()
# Select digits
index1, index2 = 0, 1  
# images 2 bytes
image1_bytes = BytesIO()
image2_bytes = BytesIO()
plt.imsave(image1_bytes, digits.images[index1], cmap='gray', format='png')
plt.imsave(image2_bytes, digits.images[index2], cmap='gray', format='png')
image1_bytes.seek(0)
image2_bytes.seek(0)
#request
url = 'http://localhost:5000/compare_digits'
files = {
    'image1': ('image1.png', image1_bytes, 'image/png'),
    'image2': ('image2.png', image2_bytes, 'image/png')
}
# Send request
response = requests.post(url, files=files)
# Process  response
if response.status_code == 200:
    print(response.json())
else:
    print('failed to get a response from server:', response.status_code)