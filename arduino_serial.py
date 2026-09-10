import serial
import time

# ==========================================
# Arduino COM Port
# ==========================================

PORT = "COM12"       # CHANGE THIS
BAUD_RATE = 9600


# ==========================================
# Connect to Arduino
# ==========================================

print("Connecting to Arduino...")

arduino = serial.Serial(
    PORT,
    BAUD_RATE,
    timeout=2
)

time.sleep(2)

print("Arduino connected!")
print("Waiting for sensor data...\n")


# ==========================================
# Read Arduino continuously
# ==========================================

while True:

    try:

        line = arduino.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if line:

            print(line)

    except KeyboardInterrupt:

        print("\nStopping...")

        arduino.close()

        break

    except Exception as e:

        print("Error:", e)

        break