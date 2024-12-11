import tkinter as tk
import firebase_admin
from firebase_admin import credentials, messaging
import os
import subprocess
import json
from paddleocr import PaddleOCR
from googletrans import Translator
from threading import Thread
import RPi.GPIO as GPIO
import time
import cv2
import numpy as np

# Firebase Admin 초기화
cred = credentials.Certificate("/home/pompompurin/translight/serviceAccountKey.json")  # 서비스 계정 키 파일 경로
firebase_admin.initialize_app(cred)

# 설정
BUTTON_PIN = 24
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "tmp", "image.jpg")

# PaddleOCR Lite 초기화
ocr = PaddleOCR(
    use_gpu=False,
    lang='en',
)

translator = Translator()

# GPIO 설정
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Tkinter 화면 설정
root = tk.Tk()
root.title("Translight & Timer Control")
root.geometry("800x600")
root.configure(bg="white")

# Main Frame
top_frame = tk.Frame(root)
top_frame.pack(side="top", fill="both", expand=True)

# Timer Control Section
timer_frame = tk.Frame(top_frame, width=100, bg="lightgrey")
timer_frame.pack(side="left", fill="y")

start_btn = tk.Button(timer_frame, text="Start Timer", command=lambda: send_timer("start"), height=2, width=12)
start_btn.pack(pady=20)

stop_btn = tk.Button(timer_frame, text="Stop Timer", command=lambda: send_timer("stop"), height=2, width=12)
stop_btn.pack(pady=20)

# Highlight OCR Section
content_frame = tk.Frame(top_frame)
content_frame.pack(side="right", expand=True, fill="both")

status_label = tk.Label(content_frame, text="버튼을 눌러 작업을 시작하세요.", font=("Arial", 16))
status_label.pack(pady=10)

text_frame = tk.Frame(content_frame)
text_frame.pack(expand=True, fill="both", pady=20)

original_text_box = tk.Text(text_frame, width=32, height=16, state="disabled", wrap="word", font=("Arial", 16))
original_text_box.grid(row=0, column=0, padx=10)

translated_text_box = tk.Text(text_frame, width=32, height=16, state="disabled", wrap="word", font=("Arial", 16))
translated_text_box.grid(row=0, column=1, padx=10)

trans_history = []

# Firebase 메시지 전송 함수
def send_message(topic, action, additional_data=None):
    data = {"action": action}
    if additional_data:
        data.update(additional_data)
    message = messaging.Message(data=data, topic=topic)
    response = messaging.send(message)
    print(f"Successfully sent message: {action} to topic {topic}")

def send_timer(action):
    send_message("timer", action)

# libcamera-still로 이미지 캡처
def capture_image(image_path):
    try:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        subprocess.run(
            ["libcamera-still", "-o", image_path, "--timeout", "1000"],
            check=True
        )
        print(f"이미지가 {image_path}에 저장되었습니다.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"이미지 캡처 오류: {e}")
        return False

# 노란 형광펜 영역 추출 및 OCR
def process_highlighted_text(image_path):
    try:
        # 이미지 읽기
        image = cv2.imread(image_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # HSV 범위 설정 (노란색 확장)
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([35, 255, 255])

        # 대비 향상 (CLAHE 사용)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        hsv = cv2.merge((h, s, v))

        # 노란색 마스크 생성
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 마스크 확장 (팽창 + 침식)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)

        # 마스크 적용
        result = cv2.bitwise_and(image, image, mask=mask)

        # 결과 이미지 저장
        temp_image_path = os.path.join(BASE_DIR, "tmp", "highlighted.jpg")
        cv2.imwrite(temp_image_path, result)

        # OCR 수행
        result = ocr.ocr(temp_image_path, cls=False)
        text = "\n".join([line[1][0] for line in result[0]])
        print("노란 형광펜 OCR 결과:", text)
        return text
    except Exception as e:
        print(f"노란 형광펜 처리 오류: {e}")
        return ""

# 번역 처리
def translate_text(text):
    try:
        translated = translator.translate(text, src="en", dest="ko").text
        print("번역 결과:", translated)
        return translated
    except Exception as e:
        print(f"번역 오류: {e}")
        return "번역 실패"

# 번역 결과 저장 (딕셔너리 형식)
def save_translation_as_dict(original, translated):
    try:
        print(f'저장 완료: "{original}" => "{translated}"')
    except Exception as e:
        print(f"딕셔너리 저장 오류: {e}")

# 버튼 대기 함수
def wait_for_button_press():
    print("버튼을 눌러 작업을 시작하세요.")
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.HIGH:
            print("버튼이 눌렸습니다!")
            status_label.config(text="촬영 중...")
            root.update()

            if not capture_image(IMAGE_PATH):
                status_label.config(text="이미지 캡처 실패")
                root.update()
                continue

            ocr_text = process_highlighted_text(IMAGE_PATH)
            if not ocr_text.strip():
                status_label.config(text="OCR 실패")
                root.update()
                continue

            status_label.config(text="번역 중...")
            root.update()
            translated_text = translate_text(ocr_text)

            # Firebase로 메시지 전송
            send_message("words", "highlight", {"word": ocr_text, "meaning": translated_text})

            # Tkinter 화면 업데이트
            save_translation_as_dict(ocr_text, translated_text)
            display_out(ocr_text, translated_text)

            status_label.config(text="버튼을 눌러 작업을 시작하세요.")
            root.update()

# 디스플레이 업데이트
def display_out(original, translated):
    trans_history.insert(0, (original, translated))
    if len(trans_history) > 3:
        trans_history.pop()

    original_text_box.config(state="normal")
    translated_text_box.config(state="normal")
    original_text_box.delete("1.0", tk.END)
    translated_text_box.delete("1.0", tk.END)

    for i, (orig, trans) in enumerate(trans_history):
        original_text_box.insert(tk.END, f"{i+1}. {orig}\n")
        translated_text_box.insert(tk.END, f"{i+1}. {trans}\n")

    original_text_box.config(state="disabled")
    translated_text_box.config(state="disabled")
    status_label.config(text="버튼을 눌러 작업을 시작하세요.")
    root.update()

# 디스플레이 종료 함수
def on_closing():
    GPIO.cleanup()
    root.destroy()

# 메인 실행 함수
def main():
    root.protocol("WM_DELETE_WINDOW", on_closing)  # 창 닫기 이벤트 처리

    try:
        button_thread = Thread(target=wait_for_button_press, daemon=True)
        button_thread.start()
        root.mainloop()
    except KeyboardInterrupt:
        print("프로그램 종료.")
    finally:
        GPIO.cleanup()
        on_closing()

if __name__ == "__main__":
    main()
