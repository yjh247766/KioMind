import gpiod
import time
from gpiod.line import Direction, Value

CHIP_PATH = "/dev/gpiochip1"
TRIG_PIN = 4   # gpio01_A4
ECHO_PIN = 8   # gpio01_B0

chip = gpiod.Chip(CHIP_PATH)

# TRIG 핀: 출력 설정
trig_settings = gpiod.LineSettings()
trig_settings.direction = Direction.OUTPUT
trig = chip.request_lines(
    consumer="ultrasonic",
    config={TRIG_PIN: trig_settings}
)

# ECHO 핀: 입력 설정
echo_settings = gpiod.LineSettings()
echo_settings.direction = Direction.INPUT
echo = chip.request_lines(
    consumer="ultrasonic",
    config={ECHO_PIN: echo_settings}
)

print("▶ 초음파 거리 측정 시작")

try:
    while True:
        # 트리거 신호 생성 (LOW -> HIGH 20μs -> LOW)
        trig.set_values({TRIG_PIN: Value.INACTIVE})
        time.sleep(0.000002)
        trig.set_values({TRIG_PIN: Value.ACTIVE})
        time.sleep(0.00002)  # 20μs로 펄스 길이 조정
        trig.set_values({TRIG_PIN: Value.INACTIVE})

        # Echo HIGH 대기 (최대 25ms)
        timeout = time.time() + 0.01
        while echo.get_value(ECHO_PIN) == Value.INACTIVE:
            if time.time() > timeout:
                print("❌ Echo 응답 없음")
                break
        else:
            start = time.time()

            # Echo LOW 대기 (최대 25ms)
            timeout = time.time() + 0.01
            while echo.get_value(ECHO_PIN) == Value.ACTIVE:
                if time.time() > timeout:
                    print("⚠️ Echo HIGH 유지 초과")
                    break
            end = time.time()

            duration = end - start
            distance = duration * 34300 / 2  # cm 단위 거리 계산
            print(f"📏 거리: {distance:.2f} cm")
            if(distance <= 60):
                print("true")
            else:
                print("false")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n⏹ 측정 종료")
