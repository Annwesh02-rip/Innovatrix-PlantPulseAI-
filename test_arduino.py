import serial
import time

arduino = serial.Serial("COM11", 9600)

time.sleep(2)

while True:
    data = arduino.readline().decode("utf-8").strip()
    print(data)