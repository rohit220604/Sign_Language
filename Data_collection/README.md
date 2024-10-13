# Sign Language Detection

# Overview

This project focuses on sign language detection using computer vision techniques. The main goal is to capture and process hand gestures to recognize various sign language symbols. This implementation utilizes OpenCV and the cvzone library to detect and track hand movements.

# Technologies Used

Python, OpenCV, cvzone, NumPy

# Features
Real-time hand detection and tracking   
Cropping and resizing of hand gesture images  
Saving processed images for further use

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SignLanguageDetection.git
```
2. Navigate to the project directory:
```bash
cd SignLanguageDetection
```
3. Install the required dependencies:
```bash
pip install opencv-python cvzone numpy
```

## Usage

To run the sign language detection script, use the following command:
```bash
python sign_language_detection.py
```
Press the 's' key to save the detected hand gesture image. The images will be saved in the specified folder with a timestamp.

```python
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder = "/Users/dnyan/OneDrive/Desktop/SLD/data/ok"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap: wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
```

## Contributing

1. Fork the repository.

2. Create a new branch:
```bash
git checkout -b feature/your-feature
```

3. Make your changes and commit them:
```bash
git commit -m 'Add some feature'
```

4. Push to the branch:
```bash
git push origin feature/your-feature
```

5. Open a pull request.


## License

[MIT](https://choosealicense.com/licenses/mit/)