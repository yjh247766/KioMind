import gpiod
import time
from gpiod.line import Direction, Value

# GPIO chip and pin configuration for HC-SR04
CHIP_PATH = "/dev/gpiochip1"
TRIG_PIN = 4   # gpio01_A4
ECHO_PIN = 8   # gpio01_B0

def measure_distance(threshold_cm=100) -> bool:
    """
    Trigger the HC-SR04 ultrasonic sensor and measure the distance to the nearest object.

    Opens the GPIO chip on every call. For production use at high polling rates,
    consider refactoring into a class that holds the chip and line handles open persistently.

    Args:
        threshold_cm: distance threshold in centimeters (default: 100 cm)

    Returns:
        True if an object is detected within threshold_cm, False otherwise.
        Also returns False on echo timeout (no object in range).
    """
    # Open the GPIO chip
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

    # Send a 20 µs trigger pulse: LOW -> HIGH -> LOW
    trig.set_values({TRIG_PIN: Value.INACTIVE})
    time.sleep(0.000002)
    trig.set_values({TRIG_PIN: Value.ACTIVE})
    time.sleep(0.00002)   # 20 µs pulse
    trig.set_values({TRIG_PIN: Value.INACTIVE})

    # Wait for ECHO pin to go HIGH (start of return pulse), with 10 ms timeout
    timeout = time.time() + 0.01
    while echo.get_value(ECHO_PIN) == Value.INACTIVE:
        if time.time() > timeout:
            return False  # No echo received
    start = time.time()

    # Wait for ECHO pin to go LOW (end of return pulse), with 10 ms timeout
    timeout = time.time() + 0.01
    while echo.get_value(ECHO_PIN) == Value.ACTIVE:
        if time.time() > timeout:
            return False  # Echo held too long (object too close or error)
    end = time.time()

    # Calculate distance: speed of sound = 343 m/s = 34300 cm/s
    # Divide by 2 for round-trip
    duration = end - start
    distance = duration * 34300 / 2

    return distance <= threshold_cm
