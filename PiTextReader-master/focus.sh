#!/bin/bash
# Utility to help with lighting and focusing camera
# Use a browser and navigate to the URL displayed 

cp focus.html /tmp
cd /tmp

# Ctrl-C 처리 함수
ctrl_c() {
    echo "** Trapped CTRL-C"
    kill $!
    exit $?
}

trap ctrl_c INT

# Run a web server to view the photo
IP=`hostname -I`
IP=${IP%?}
echo "To see photo, browse to http://$IP:8080/focus.html"
echo "Press Ctrl-C to exit"
echo 
echo
# Python3의 http.server로 웹 서버 실행
python3 -m http.server 8080 &

# 웹 서버가 실행될 시간을 줌
sleep 5

# 무한 루프에서 계속 사진 촬영
while true; do
    echo "taking photo"
    raspistill -cfx 128:128 -w 1024 -h 768 --awb auto -rot 180 -t 1500 -o image.jpg
done


