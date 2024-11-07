#!/bin/bash
# Install PiTextReader with PaddleOCR
# 
# Run using:
# $ sh install.sh
#
# Can be safely run multiple times
#
# version 20231107
#

# Make sure python requirements are installed
sudo apt-get update

# Install necessary packages
# flite는 TTS(Text-to-Speech)용 패키지입니다.
sudo apt-get install -y flite

# Install Python and pip if not already installed
# 파이썬과 pip 설치 (라즈베리파이에 파이썬3가 기본 설치되어 있지 않은 경우)
sudo apt-get install -y python3 python3-pip

# Install PaddleOCR and PaddlePaddle
echo "Installing PaddleOCR and PaddlePaddle..."
pip3 install paddlepaddle
pip3 install paddleocr

# Verify Camera is configured
X=`raspistill -o test.jpg 2>&1|grep Failed`

if [ -z "$X" ];
then
    echo "Found Camera OK"
else
    echo $X
    echo "NO Camera Detected! SEE DOCS Troubleshooting section."
    exit
fi 

# Install custom software (crontab 설정)
crontab ./cronfile
echo "Crontab entry installed for pi userid. OK"

# FINISHED!
echo "Finished installation. See Readme.md for more info"
echo "Reboot your pi now: $ sudo reboot"

