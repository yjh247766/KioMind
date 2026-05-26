import gpiod
import time
from gpiod.line import Direction, Value

# Standalone ultrasonic sensor test using the gpiod (libgpiod v2) library.
# GPIO chip and pin configuration
CHIP_PATH = "/dev/gpiochip1"
TRIG_PIN = 4   # gpio01_A4
ECHO_PIN = 8   # gpio01_B0

# Open chip and request lines once at module level for continuous polling
chip = gpiod.Chip(CHIP_PATH)

# Configure TRIG pin as output
trig_settings = gpiod.LineSettings()
trig_settings.direction = Direction.OUTPUT
trig = chip.request_lines(
    consumer="ultrasonic",
    config={TRIG_PIN: trig_settings}
)

# Configure ECHO pin as input
echo_settings = gpiod.LineSettings()
echo_settings.direction = Direction.INPUT
echo = chip.request_lines(
    consumer="ultrasonic",
    config={ECHO_PIN: echo_settings}
)

print("Ultrasonic distance measurement starting")

try:
    while True:
        # Generate trigger pulse: LOW -> HIGH (20 µs) -> LOW
        trig.set_values({TRIG_PIN: Value.INACTIVE})
        time.sleep(0.000002)
        trig.set_values({TRIG_PIN: Value.ACTIVE})
        time.sleep(0.00002)   # 20 µs pulse width
        trig.set_values({TRIG_PIN: Value.INACTIVE})

        # Wait for ECHO to go HIGH, with 10 ms timeout
        timeout = time.time() + 0.01
        while echo.get_value(ECHO_PIN) == Value.INACTIVE:
            if time.time() > timeout:
                print("No echo received")
                break
        else:
            start = time.time()

            # Wait for ECHO to go LOW, with 10 ms timeout
            timeout = time.time() + 0.01
            while echo.get_value(ECHO_PIN) == Value.ACTIVE:
                if time.time() > timeout:
                    print("Echo HIGH held too long")
                    break
            end = time.time()

            # Calculate distance: speed of sound = 34300 cm/s, divide by 2 for round-trip
            duration = end - start
            distance = duration * 34300 / 2
            print(f"Distance: {distance:.2f} cm")

            if distance <= 60:
                print("true")
            else:
                print("false")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nMeasurement stopped")
