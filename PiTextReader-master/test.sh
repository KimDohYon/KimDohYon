#!/bin/bash
# TEST RASPI with PaddleOCR

# CLEANUP
rm -f test.jpg test.txt

# ADJUST AUDIO VOLUME
# Change VOLUME to increase/decrease 0-100%
VOLUME=90%
echo "Setting volume to $VOLUME"
sudo amixer -q sset PCM,0 ${VOLUME}

# PLAY SPEECH 
echo "playing TTS"
flite -voice awb -t "Hello World! This is a test of text to speech."

# TEST CAMERA
echo "taking photo"
raspistill -cfx 128:128 --awb auto -rot 180 -t 500 -o test.jpg
ls -l test.jpg

# OCR test using PaddleOCR
echo "Converting to Text using PaddleOCR, standby..."

# PaddleOCR을 실행하기 위해 Python 스크립트를 사용합니다.
# 이 부분에서 PaddleOCR을 통해 OCR을 수행하고 결과를 test.txt에 저장하도록 합니다.

python3 <<EOF
from paddleocr import PaddleOCR
import sys

# PaddleOCR 객체 생성
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # 언어를 변경하려면 'en' 대신 'korean' 등 사용

# OCR 수행
result = ocr.ocr('test.jpg', cls=True)
extracted_text = '\n'.join([line[1][0] for line in result[0]])

# 결과를 test.txt 파일로 저장
with open("test.txt", "w") as f:
    f.write(extracted_text)

print("OCR 결과:")
print(extracted_text)
EOF

# OCR 결과 출력
cat test.txt

# SPEAK TEXT
flite -voice awb -f test.txt

# Run a web server to view photo
IP=`hostname -I`
IP=${IP%?}
echo "To see photo, browse to http://$IP:8080/test.jpg"
echo "Press Ctrl-C to exit"
# Python3의 http.server 사용
python3 -m http.server 8080


