import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator
from scipy.fftpack import fft
import struct
import serial.tools.list_ports

# # 列出可用串口
# ports = list(serial.tools.list_ports.comports())
# for i, port in enumerate(ports):
#     print(f"{i}: {port.device} - {port.description}")

# # 選擇串口
# port_idx = input("選擇串口號碼 (默認0): ") or "0"
# port = ports[int(port_idx)].device

# 配置參數
SAMPLE_SIZE = 5000            # 採樣點數
ADC_CLOCK = 20_000_000  # ADC時鐘頻率
SAMPLE_RATE = ADC_CLOCK / 20   # 採樣率
header = b'\xAA\x55'         # 幀頭
footer = b'\x5A\xA5'         # 幀尾

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
ax1.set_title('Time Domain Signal')
ax1.set_xlabel('Sample Number')
ax1.set_ylabel('Voltage (V)')
ax1.set_ylim(-1.7, 1.7)
ax1.grid(True, alpha=0.3)
time_line, = ax1.plot([], [], 'c-', linewidth=1)

# 設置頻域圖表
ax2.set_title('Frequency Domain Signal')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Amplitude')
# ax2.set_ylim(-60, 40)
ax2.set_ylim(0, 1200)
# ax2.set_xlim(0, 200_000)
# ax2.xaxis.set_major_locator(MultipleLocator(1e4))
ax2.grid(True, alpha=0.3)
freq_line, = ax2.plot([], [], 'r-', linewidth=1)

# 創建x軸數據
time_x = np.arange(SAMPLE_SIZE)
freq_x = np.fft.rfftfreq(SAMPLE_SIZE, 1/SAMPLE_RATE)

# 初始化y軸數據
time_y = np.zeros(SAMPLE_SIZE)
freq_y = np.zeros(len(freq_x))

# 數據幀計數
frame_counter = 0
sampling_count = 0

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

def init():
    time_line.set_data(time_x, time_y)
    freq_line.set_data(freq_x, freq_y)
    ax1.set_xlim(0, SAMPLE_SIZE)
    ax2.set_xlim(0, SAMPLE_RATE/2)
    return time_line, freq_line

def update(frame):
    global time_y, freq_y, sampling_count
    
    try:
        # 確保有足夠數據可讀
        if ser.in_waiting > 0:
            # 嘗試讀取一個完整數據幀
            data = align_and_get_data(ser)
            
            if data is not None and len(data) == SAMPLE_SIZE:
                # 更新時域數據
                data = data * 3.3 / 4095
                data = data - np.median(data)
                time_y = data
                time_line.set_data(time_x, time_y)
                
                # 計算FFT
                windowed_data = time_y * np.hamming(len(time_y))
                fft_data = np.fft.rfft(windowed_data)
                
                # 計算幅度譜 (dB)
                # freq_y = 20 * np.log10(np.abs(fft_data) / SAMPLE_SIZE + 1e-10)
                freq_y = np.abs(fft_data)
                # freq_y = freq_y / np.max(freq_y)
                
                # 更新頻域圖
                freq_line.set_data(freq_x, freq_y)
                

                
                # 打印統計信息
                print(f"ADC數據範圍: {np.min(data)} - {np.max(data)}")
            
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()
    
    return time_line, freq_line

# 創建動畫
ani = FuncAnimation(fig, update, frames=None, init_func=init, 
                    interval=50, blit=True, cache_frame_data=False)

try:
    plt.tight_layout()
    plt.show()
except KeyboardInterrupt:
    print("程序被中斷")
finally:
    ser.close()
    print("串口已關閉")