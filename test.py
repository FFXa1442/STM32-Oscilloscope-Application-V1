from serial import Serial
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import asyncio
import threading
import queue
import csv
import os
from datetime import datetime

filepath_ = os.path.join(os.path.dirname(__file__), 'data')
if os.path.exists(filepath_) is False:
    os.makedirs(filepath_)

def align_and_get_data(ser: Serial):
    """
    This function will read bytes from the serial port until it finds a specific start sequence (0xFF, 0xFA)
    and then continues to read until it finds an end sequence (0xFF, 0xFB).
    Args:
        ser (Serial): MCU Device Serial Object

    Returns:
        data (bytes): Data received from the serial device
    """
    while True:
        if ser.read(1) != b'\xFF':
            continue
        if ser.read(1) != b'\xFA':
            continue
        break
    return ser.read_until(b'\xFF\xFB')[:-2]

# Queue for passing data between threads
data_queue = queue.Queue()

# Function to save data to CSV
async def save_to_csv(data, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # print(timestamp)
        filename = f"data_{timestamp}.csv"
    
    filepath = os.path.join(os.path.join(os.path.dirname(__file__), 'data'), filename)
    
    # Open file and write data asynchronously
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['index', 'value'])  # Write header
        
        for i, value in enumerate(data):
            writer.writerow([i, float(value)])
    
    print(f"Data saved to {filepath}")

# Async worker function
async def csv_worker():
    while True:
        try:
            if not data_queue.empty():
                data = data_queue.get()
                await save_to_csv(data)
                data_queue.task_done()
            else:
                await asyncio.sleep(0.0001)
        except Exception as e:
            print(f"Error in CSV worker: {e}")
            await asyncio.sleep(1)

# Function to run the async worker in a separate thread
def run_async_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(csv_worker())
    finally:
        loop.close()

# Start the worker thread
worker_thread = threading.Thread(target=run_async_worker, daemon=True)
worker_thread.start()

ser = Serial('COM3', 115200)  # Adjust COM port and baud rate as needed

# based on uint16_t
max_len = 5000  # Maximum length of the deque

try:
    while True:
        data = align_and_get_data(ser)
        data = np.frombuffer(data, dtype=np.uint8)
        # print(data[:10])  
        if data.shape[0] != max_len * 2:
            print("Data length mismatch:", data.shape[0])
            continue
        
        values = data.view(np.uint16)
        values = values * 3.3 / 4095
        values *= 1000.0

        # Queue the data for async saving
        data_queue.put(values.copy())

except KeyboardInterrupt:
    print("Exiting...")
    ser.close()
    # Wait for remaining tasks to complete
    data_queue.join()

