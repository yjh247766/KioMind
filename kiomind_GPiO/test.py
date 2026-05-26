import OPi.GPIO as GPIO
import time

# Standalone ultrasonic sensor test using the OPi.GPIO library.
# Uses physical (BOARD) pin numbering.
TRIG = 18
ECHO = 22

GPIO.setmode(GPIO.BOARD)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

print("Ultrasonic distance measurement starting")

try:
    while True:
        # Generate a short trigger pulse: LOW -> HIGH (10 µs) -> LOW
        GPIO.output(TRIG, GPIO.LOW)
        time.sleep(0.0002)
        GPIO.output(TRIG, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(TRIG, GPIO.LOW)

        # Measure the duration of the echo pulse
        while GPIO.input(ECHO) == GPIO.LOW:
            pulse_start = time.time()

        while GPIO.input(ECHO) == GPIO.HIGH:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 34300 / 2  # Round-trip time -> cm

        print(f"Distance: {distance:.2f} cm")
        time.sleep(1)

except KeyboardInterrupt:
    print("\nMeasurement stopped")
    GPIO.cleanup()
