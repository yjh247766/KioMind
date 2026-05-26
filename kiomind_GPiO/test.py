import  OPi.GPIO  as GPIO
import time

TRIG = 18  # 물리적 핀 번호
ECHO = 22  # 물리적 핀 번호

GPIO.setmode(GPIO.BOARD)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

print("초음파 센서 거리 측정 시작")

try:
    while True:
        # Trig 핀에 짧은 펄스 발생
        GPIO.output(TRIG, GPIO.LOW)
        time.sleep(0.0002)
        GPIO.output(TRIG, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(TRIG, GPIO.LOW)

        # Echo 핀에서 응답 시간 측정
        while GPIO.input(ECHO) == GPIO.LOW:
            pulse_start = time.time()

        while GPIO.input(ECHO) == GPIO.HIGH:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 34300 / 2  # 왕복 거리 → cm

        print(f"거리: {distance:.2f} cm")
        time.sleep(1)

except KeyboardInterrupt:
    print("\n측정 종료")
    GPIO.cleanup()
