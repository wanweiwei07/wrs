import time
import serial

ser = serial.Serial(port="COM6", rtscts=True)
# ser.rtscts=True
while True:
    print(ser.cts)
    # time.sleep(.1)