import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import struct
import threading
import queue
import os
import csv
from datetime import datetime
import time
from matplotlib.widgets import TextBox

# 配置參數
PORT = 'COM3'  # 串口號
BAUDRATE = 921600  # 波特率
SAMPLE_SIZE = 5000            # 採樣點數
VOLTAGE_REF = 3.3 * 2  # 參考電壓
RESOLUTION = 4095  # ADC分辨率
ADC_CLOCK = 20_000_000  # ADC時鐘頻率
SAMPLE_RATE = ADC_CLOCK / 20   # 採樣率
RANGE = int(SAMPLE_RATE / 2)               # 頻域顯示範圍
FREQ_DOMAIN_MAX_ENERGY = 1000  # 頻域最大能量值
# RANGE = 50_000               # 頻域顯示範圍
ENABLE_TIME_DOMAIN_NORMALIZATION = False  # 是否啟用時域歸一化

# 頻域顯示選項（簡化，只允許一種模式啟用）
FREQ_MODE = None  # 'normal', 'db', 'db_norm'

header = b'\xAA\x55'         # 幀頭
footer = b'\x5A\xA5'         # 幀尾

# 數據存儲配置
def get_today():
    return datetime.now().strftime("%Y%m%d")

MAIN_DIR = get_today()
DATA_TYPE = "Object"
DATA_INDEX = "0"

def build_save_directory(main_dir, data_type, data_index):
    return f"{main_dir}/captured_data_{data_type}_collection_{data_index}"

SAVE_DIRECTORY = build_save_directory(MAIN_DIR, DATA_TYPE, DATA_INDEX)

def unit_to_dB(value):
    return 10 * np.log10(value + 1e-12)  # 避免對0取對數

def directory_init():
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)
        
def send_start_id(ser):
    """
    向STM32發送啟動ID，只有收到此ID才會傳送數據
    """
    start_id = 214802728
    ser.write(start_id.to_bytes(4, byteorder='little'))
    print(f"已發送啟動ID: {start_id}")

stm32 = serial.Serial(PORT, BAUDRATE, timeout=1)
print(f"串口已打開: {stm32.name}")

# 清空緩衝區
stm32.reset_input_buffer()

# 創建圖表
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.tight_layout(pad=3.0)

# 設置時域圖表
ax1.set_title(f'Time Domain{' (Normalization)' if ENABLE_TIME_DOMAIN_NORMALIZATION else ''}')
# ax1.set_xlabel('Sample Number')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Voltage (V)')

ax1_ylim = { 'bottom': -1.1, 'top': 1.1, } if ENABLE_TIME_DOMAIN_NORMALIZATION else { 'bottom': -VOLTAGE_REF, 'top': VOLTAGE_REF, }
ax1.set_ylim(**ax1_ylim)

ax1.grid(True, alpha=0.3)
time_line, = ax1.plot([], [], 'c-', linewidth=1)

# 設置頻域圖表
if FREQ_MODE == 'normal':
    ax2.set_title('Frequency Domain (Normalization)')
    ax2.set_ylabel('Energy')
    ax2.set_ylim(-1e-2, 1.1)
elif FREQ_MODE == 'db':
    ax2.set_title('Frequency Domain')
    ax2.set_ylabel('Energy (dB)')
    ax2.set_ylim(np.floor(unit_to_dB(0)), np.ceil(unit_to_dB(FREQ_DOMAIN_MAX_ENERGY)))
elif FREQ_MODE == 'db_norm':
    ax2.set_title('Frequency Domain (Normalization)')
    ax2.set_ylabel('Energy (dB)')
    ax2.set_ylim(-1.1, 1.1)
else:
    ax2.set_title('Frequency Domain')
    ax2.set_ylabel('Energy')
    ax2.set_ylim(-1, FREQ_DOMAIN_MAX_ENERGY)

ax2.set_xlabel('Frequency (kHz)')
ax2.grid(True, alpha=0.3)
freq_line, = ax2.plot([], [], 'g-', linewidth=1)

# 創建x軸數據
# time_x = np.arange(SAMPLE_SIZE)
time_x = np.linspace(0, SAMPLE_SIZE/SAMPLE_RATE, SAMPLE_SIZE) * 1000
freq_x = np.fft.rfftfreq(SAMPLE_SIZE, 1/SAMPLE_RATE) / 1000


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
            if stm32.in_waiting > 0 or True:
                
                # 每次都先發送啟動ID
                send_start_id(stm32)
                # 嘗試讀取一個完整數據幀
                data = align_and_get_data(stm32)
                
                if data is not None and len(data) == SAMPLE_SIZE:
                    # 轉換為電壓值 (ADC值 * 參考電壓 / ADC分辨率)
                    data_volts = data * VOLTAGE_REF / RESOLUTION
                    # data_centered = data_volts - np.median(data_volts)
                    data_centered = data_volts - np.mean(data_volts)  # 使用均值去除直流分量
                    
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
        time.sleep(1 / 1000)

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
                    base_filename = f"{SAVE_DIRECTORY}/{DATA_TYPE}.{timestamp}.{file_counter}"
                    
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
        base_filename = f"{SAVE_DIRECTORY}/{DATA_TYPE}.{timestamp}.{file_counter}"
        
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
    ax1.set_xlim(0, (SAMPLE_SIZE/SAMPLE_RATE) * 1000)
    ax2.set_xlim(0, RANGE / 1000)
    return time_line, freq_line

def update(frame):
    global time_y, freq_y
    
    try:
        
        # 計算FFT
        freq_y = np.abs(np.fft.rfft(time_y * np.hamming(len(time_y))))
        if FREQ_MODE == 'normal':
            freq_y = freq_y / np.max(np.abs(freq_y))
        elif FREQ_MODE == 'db':
            freq_y = unit_to_dB(freq_y)
        elif FREQ_MODE == 'db_norm':
            freq_y = unit_to_dB(freq_y)
            freq_y = freq_y / np.max(np.abs(freq_y))
            
        # 更新頻域圖
        freq_line.set_data(freq_x, freq_y)
        
        # 打印統計信息
        if frame % 20 == 0:  # 減少打印頻率
            print(f"ADC數據範圍: {np.min(time_y)} ~ {np.max(time_y)}")
            
        if ENABLE_TIME_DOMAIN_NORMALIZATION:
            time_y = time_y / np.max(np.abs(time_y))
            
        # 使用當前的time_y數據更新圖表
        time_line.set_data(time_x, time_y)
        
    except Exception as e:
        print(f"更新圖表錯誤: {e}")
        import traceback
        traceback.print_exc()
    
    return time_line, freq_line

# 創建動畫
ani = FuncAnimation(fig, update, frames=None, init_func=init, 
                    interval=50, blit=True, cache_frame_data=False)

# 創建一個單獨的控制面板視窗
control_fig = plt.figure(figsize=(4, 6))
control_fig.canvas.manager.set_window_title('控制面板')

# row1: Main Directory Label
main_dir_label_ax = control_fig.add_axes([0.05, 0.88, 0.4, 0.05])
main_dir_label_ax.axis('off')
main_dir_label = main_dir_label_ax.text(0, 0.5, 'Main Directory', color='white', ha='left', va='center', fontsize=10)

# row2: Main Directory TextBox
main_dir_ax = control_fig.add_axes([0.05, 0.82, 0.9, 0.06])
main_dir_box = TextBox(main_dir_ax, '', initial=MAIN_DIR)
main_dir_box.text_disp.set_color('white')
main_dir_box.text_disp.set_backgroundcolor('#222222')

# row3: Type Label
type_label_ax = control_fig.add_axes([0.05, 0.76, 0.4, 0.05])
type_label_ax.axis('off')
type_label = type_label_ax.text(0, 0.5, 'Type', color='white', ha='left', va='center', fontsize=10)

# row4: Type TextBox
type_ax = control_fig.add_axes([0.05, 0.70, 0.9, 0.06])
type_box = TextBox(type_ax, '', initial=DATA_TYPE)
type_box.text_disp.set_color('white')
type_box.text_disp.set_backgroundcolor('#222222')

# row5: Index Label
index_label_ax = control_fig.add_axes([0.05, 0.64, 0.4, 0.05])
index_label_ax.axis('off')
index_label = index_label_ax.text(0, 0.5, 'Index', color='white', ha='left', va='center', fontsize=10)

# row6: Index TextBox
index_ax = control_fig.add_axes([0.05, 0.58, 0.9, 0.06])
index_box = TextBox(index_ax, '', initial=DATA_INDEX)
index_box.text_disp.set_color('white')
index_box.text_disp.set_backgroundcolor('#222222')

# row7: Save Dir Label
save_dir_label_ax = control_fig.add_axes([0.05, 0.52, 0.4, 0.05])
save_dir_label_ax.axis('off')
save_dir_label = save_dir_label_ax.text(0, 0.5, 'Save Dir', color='white', ha='left', va='center', fontsize=10)

# row8: Save Dir TextBox (read-only)
save_dir_ax = control_fig.add_axes([0.05, 0.46, 0.9, 0.06])
save_dir_box = TextBox(save_dir_ax, '', initial=SAVE_DIRECTORY)
save_dir_box.text_disp.set_color('white')
save_dir_box.text_disp.set_backgroundcolor('#222222')
save_dir_box.set_active(False)

def update_save_dir_from_fields(_=None):
    global MAIN_DIR, DATA_TYPE, DATA_INDEX, SAVE_DIRECTORY
    MAIN_DIR = main_dir_box.text.strip()
    DATA_TYPE = type_box.text.strip()
    DATA_INDEX = index_box.text.strip()
    SAVE_DIRECTORY = build_save_directory(MAIN_DIR, DATA_TYPE, DATA_INDEX)
    save_dir_box.set_val(SAVE_DIRECTORY)
    print(f"數據儲存路徑已更新: {SAVE_DIRECTORY}")

main_dir_box.on_submit(update_save_dir_from_fields)
type_box.on_submit(update_save_dir_from_fields)
index_box.on_submit(update_save_dir_from_fields)

# row9: Button
save_button_ax = control_fig.add_axes([0.3, 0.32, 0.4, 0.08])
save_button = plt.Button(save_button_ax, 'Start Save Data',
                         color='#3498db', hovercolor='#2980b9')
save_button.label.set_color('white')

def toggle_save_data(event):
    global save_data_enabled
    save_data_enabled = not save_data_enabled
    if save_data_enabled:
        directory_init()  # 初始化保存目錄
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
    stm32.close()
    print("串口已關閉")