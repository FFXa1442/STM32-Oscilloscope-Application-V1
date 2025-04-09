import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator
from scipy.fftpack import fft
import struct
import serial.tools.list_ports
import threading
import queue
import os
import csv
from datetime import datetime
import time

# 配置參數
SAMPLE_SIZE = 5000            # 採樣點數
ADC_CLOCK = 20_000_000  # ADC時鐘頻率
SAMPLE_RATE = ADC_CLOCK / 20   # 採樣率
OBJECT_NAME = '100kHz'
RANGE = int(SAMPLE_RATE / 2)               # 頻域顯示範圍
header = b'\xAA\x55'         # 幀頭
footer = b'\x5A\xA5'         # 幀尾

# 數據存儲配置
SAVE_DIRECTORY = "captured_data"
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)

# 建立串口連接
ser = serial.Serial('COM3', 921600, timeout=1)
print(f"串口已打開: {ser.name}")

# 清空緩衝區
ser.reset_input_buffer()

# 創建圖表
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.tight_layout(pad=3.0)

# 設置時域圖表
ax1.set_title('Time Domain')
# ax1.set_xlabel('Sample Number')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Voltage (V)')
ax1.set_ylim(-1.7, 1.7)
ax1.grid(True, alpha=0.3)
time_line, = ax1.plot([], [], 'c-', linewidth=1)

# 設置頻域圖表
ax2.set_title('Frequency Domain')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Amplitude')
ax2.set_ylim(0, 1200)
ax2.grid(True, alpha=0.3)
freq_line, = ax2.plot([], [], 'r-', linewidth=1)

# 創建x軸數據
# time_x = np.arange(SAMPLE_SIZE)
time_x = np.linspace(0, SAMPLE_SIZE/SAMPLE_RATE, SAMPLE_SIZE)
freq_x = np.fft.rfftfreq(SAMPLE_SIZE, 1/SAMPLE_RATE)[:RANGE]

# 初始化y軸數據
time_y = np.zeros(SAMPLE_SIZE)
freq_y = np.zeros(len(freq_x))

# 數據幀計數
frame_counter = 0
sampling_count = 0

# 數據存儲控制變量
is_running = True
save_data_enabled = False  # 默認不保存CSV
data_queue = queue.Queue(maxsize=100)

# 對齊數據並獲取
def align_and_get_data(ser):
    # 先讀取直到找到幀頭
    dd = ser.read_until(header)[-2:]
    while dd != header:
        print("等待幀頭")
        dd = ser.read_until(header)[-2:]
    
    print("找到幀頭")
    
    # 接著讀取長度信息（2個字節）
    length_bytes = ser.read(2)
    if len(length_bytes) != 2:
        print("讀取長度錯誤")
        return None
    
    length = struct.unpack('<H', length_bytes)[0]
    if length != SAMPLE_SIZE:
        print(f"長度不匹配: {length} != {SAMPLE_SIZE}")
        return None
    
    # 讀取數據部分和幀尾
    data_bytes = ser.read(length*2 + 2)
    if len(data_bytes) != length*2 + 2:
        print(f"數據不完整, 收到: {len(data_bytes)} 字節")
        return None
    
    # 檢查幀尾
    if data_bytes[-2:] != footer:
        print("幀尾錯誤")
        return None
    
    # 解析數據
    adc_values = []
    for i in range(length):
        pos = i * 2
        value = struct.unpack('<H', data_bytes[pos:pos+2])[0]
        adc_values.append(value)
    
    global frame_counter
    frame_counter += 1
    if frame_counter % 10 == 0:
        print(f"已收到 {frame_counter} 幀")
    
    return np.array(adc_values)

# 保存數據到CSV文件
def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 寫入列標題
        writer.writerow(['signal'])
        # 寫入數據，每行一個數據點
        for value in data:
            writer.writerow([value])
    print(f"數據已保存到CSV文件: {filename}")

# 數據採集線程
def data_acquisition_thread():
    display_queue = queue.Queue(maxsize=5)  # 顯示用隊列
    
    while is_running:
        try:
            if ser.in_waiting > 0:
                # 嘗試讀取一個完整數據幀
                data = align_and_get_data(ser)
                
                if data is not None and len(data) == SAMPLE_SIZE:
                    # 轉換為電壓值 (ADC值 * 參考電壓 / ADC分辨率)
                    data_volts = data * 3.3 / 4095
                    data_centered = data_volts - np.median(data_volts)
                    
                    # 將原始ADC數據放入保存隊列
                    if save_data_enabled and not data_queue.full():
                        data_queue.put(data)
                    
                    # 將處理後的數據放入顯示隊列
                    if not display_queue.full():
                        display_queue.put(data_centered)
                        
            # 提供數據給顯示函數
            global time_y
            if not display_queue.empty():
                time_y = display_queue.get()
                
        except Exception as e:
            print(f"數據採集錯誤: {e}")
            import traceback
            traceback.print_exc()
            
        # 短暫休眠以避免CPU過度使用
        time.sleep(0.001)

# 數據保存線程
def data_saving_thread():
    buffer = []
    samples_per_file = 10  # 每個文件保存的採樣幀數
    file_counter = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    while is_running:
        try:
            if save_data_enabled and not data_queue.empty():
                # 從隊列獲取數據
                data = data_queue.get()
                buffer.append(data)
                
                # 達到指定幀數後保存
                if len(buffer) >= samples_per_file:
                    # 生成文件名
                    base_filename = f"{SAVE_DIRECTORY}/{OBJECT_NAME}.{timestamp}.{file_counter}"
                    
                    # 保存為CSV
                    for i, frame in enumerate(buffer):
                        csv_filename = f"{base_filename}_frame{i}.csv"
                        save_to_csv(frame, csv_filename)
                    
                    # 同時保存為NPY格式作為備份
                    # np_filename = f"{base_filename}.npy"
                    # np.save(np_filename, np.array(buffer))
                    # print(f"已保存數據批次 {file_counter} 到文件")
                    
                    # 重置緩衝區並增加計數器
                    buffer = []
                    file_counter += 1
                    
        except Exception as e:
            print(f"數據保存錯誤: {e}")
            import traceback
            traceback.print_exc()
            
        # 短暫休眠以避免CPU過度使用
        time.sleep(0.01)
    
    # 保存剩餘的數據
    if save_data_enabled and buffer:
        base_filename = f"{SAVE_DIRECTORY}/{OBJECT_NAME}.{timestamp}.{file_counter}"
        
        # 保存為CSV
        for i, frame in enumerate(buffer):
            csv_filename = f"{base_filename}_frame{i}.csv"
            save_to_csv(frame, csv_filename)
        
        # 同時保存為NPY備份
        # np_filename = f"{base_filename}.npy"
        # np.save(np_filename, np.array(buffer))
        # print(f"已保存剩餘數據")

def init():
    time_line.set_data(time_x, time_y)
    freq_line.set_data(freq_x, freq_y)
    # ax1.set_xlim(0, SAMPLE_SIZE)
    ax1.set_xlim(0, SAMPLE_SIZE/SAMPLE_RATE)
    ax2.set_xlim(0, RANGE)
    return time_line, freq_line

def update(frame):
    global time_y, freq_y
    
    try:
        # 使用當前的time_y數據更新圖表
        time_line.set_data(time_x, time_y)
        
        # 計算FFT
        windowed_data = time_y * np.hamming(len(time_y))
        fft_data = np.fft.rfft(windowed_data)[:RANGE]
        
        # 計算幅度譜
        freq_y = np.abs(fft_data)
        
        # 更新頻域圖
        freq_line.set_data(freq_x, freq_y)
        
        # 打印統計信息
        if frame % 20 == 0:  # 減少打印頻率
            print(f"ADC數據範圍: {np.min(time_y)} - {np.max(time_y)}")
            
    except Exception as e:
        print(f"更新圖表錯誤: {e}")
        import traceback
        traceback.print_exc()
    
    return time_line, freq_line

# 創建動畫
ani = FuncAnimation(fig, update, frames=None, init_func=init, 
                    interval=50, blit=True, cache_frame_data=False)

# 創建一個單獨的控制面板視窗
control_fig = plt.figure(figsize=(4, 2))
control_fig.canvas.manager.set_window_title('控制面板')
save_button_ax = control_fig.add_axes([0.3, 0.5, 0.4, 0.3])
save_button = plt.Button(save_button_ax, 'Start Save Data',
                         color='#3498db', hovercolor='#2980b9')  # 默認为Start，因為保存默認是關閉的
save_button.label.set_color('white')

def toggle_save_data(event):
    global save_data_enabled
    save_data_enabled = not save_data_enabled
    save_button.label.set_text(f"{'Stop' if save_data_enabled else 'Start'} Save Data")
    print(f"Data Save {'Enabled' if save_data_enabled else 'Disabled'}")

save_button.on_clicked(toggle_save_data)

# 啟動數據採集和保存線程
acquisition_thread = threading.Thread(target=data_acquisition_thread, daemon=True)
saving_thread = threading.Thread(target=data_saving_thread, daemon=True)

acquisition_thread.start()
saving_thread.start()
print("數據採集和保存線程已啟動")

try:
    plt.tight_layout()
    plt.show()
except KeyboardInterrupt:
    print("程序被中斷")
finally:
    # 通知線程停止
    is_running = False
    
    # 等待線程結束
    acquisition_thread.join(timeout=1.0)
    saving_thread.join(timeout=1.0)
    
    # 關閉串口
    ser.close()
    print("串口已關閉")