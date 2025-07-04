import gpiod
import time
from gpiod.line import Direction, Value

# 사용할 GPIO 칩과 핀 번호 설정
CHIP_PATH = "/dev/gpiochip1"
TRIG_PIN = 4   # gpio01_A4
ECHO_PIN = 8   # gpio01_B0

def measure_distance(threshold_cm=100) -> bool:
    # GPIO 칩 열기
    chip = gpiod.Chip(CHIP_PATH)

    # 트리거 핀 설정 (출력)
    trig_settings = gpiod.LineSettings()
    trig_settings.direction = Direction.OUTPUT
    trig = chip.request_lines(
        consumer="ultrasonic",
        config={TRIG_PIN: trig_settings}
    )

    # 에코 핀 설정 (입력)
    echo_settings = gpiod.LineSettings()
    echo_settings.direction = Direction.INPUT
    echo = chip.request_lines(
        consumer="ultrasonic",
        config={ECHO_PIN: echo_settings}
    )

    # 초음파 트리거 펄스 (20μs)
    trig.set_values({TRIG_PIN: Value.INACTIVE})
    time.sleep(0.000002)
    trig.set_values({TRIG_PIN: Value.ACTIVE})
    time.sleep(0.00002)
    trig.set_values({TRIG_PIN: Value.INACTIVE})

    # Echo 핀 HIGH 대기
    timeout = time.time() + 0.01  # 10ms 제한
    while echo.get_value(ECHO_PIN) == Value.INACTIVE:
        if time.time() > timeout:
            return False
    start = time.time()

    # Echo 핀 LOW 대기
    timeout = time.time() + 0.01
    while echo.get_value(ECHO_PIN) == Value.ACTIVE:
        if time.time() > timeout:
            return False
    end = time.time()

    # 거리 계산 (단위: cm)
    duration = end - start
    distance = duration * 34300 / 2

    return distance <= threshold_cm
