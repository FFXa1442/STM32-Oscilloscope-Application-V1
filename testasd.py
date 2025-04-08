
from serial import Serial
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

BUFFER_SIZE = 10000

serial = Serial('COM6', 912600)

header = b'\xe0\x10'
footer = b'\xf0\x10'

TSETING_SIZE = 5000

plt.ion()
fig, ax = plt.subplots()
x = np.linspace(0, 1, TSETING_SIZE // 2)
y = deque([0] * (TSETING_SIZE // 2), maxlen=TSETING_SIZE // 2)
line, = ax.plot(y)
ax.set_ylim(0, 4095)

def align_and_get_data(ser: Serial):
    """
    This function will read bytes from the serial port until it finds a specific start sequence (0xFF, 0xFA)
    and then continues to read until it finds an end sequence (0xFF, 0xFB).
    Args:
        ser (Serial): MCU Device Serial Object

    Returns:
        data (bytes): Data received from the serial device
    """
    dd = ser.read_until(header)[-2:]
    while dd != header:
        print("Waiting for header")
        pass
    
    print("Header found")
    # while serial.in_waiting < BUFFER_SIZE:
    #     print("Waiting for buffer to fill")x
    #     print(serial.in_waiting)
    #     pass
    # print("Buffer full")
    return ser.read_until(footer)[:-2]
    # return ser.read(BUFFER_SIZE)

try:
    print("GG")
    while True:

        values = np.frombuffer(align_and_get_data(serial), dtype=np.uint8)
        values = values.view(np.int16)
        values = values[(BUFFER_SIZE - TSETING_SIZE) // 2:(BUFFER_SIZE + TSETING_SIZE) // 2]
        print(values.shape, values)
            
        y.extend(values)
        line.set_ydata(y)

        plt.draw()
        plt.pause(0.01)

except KeyboardInterrupt:
    print("Program terminated.")
    serial.close()
    plt.close()