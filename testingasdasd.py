import serial
import struct
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import threading

# 定義緩衝區大小和數據結構常量
ADC_BUFFER_SIZE = 5000  # 與 STM32 代碼中定義相同
HEADER_VALUE = 0x10e0
FOOTER_VALUE = 0x10f0
PACKET_SIZE = (ADC_BUFFER_SIZE + 2) * 2  # header + data + footer (每個 uint16_t 佔 2 字節)
SAMPLING_RATE = 100000  # 100 kHz sampling rate

class ADCDataReceiver:
    def __init__(self, port, baudrate=921600):
        self.serial_port = serial.Serial(port, baudrate)
        self.data_buffer = deque(maxlen=ADC_BUFFER_SIZE)  # 存儲最近收到的 5000 個 ADC 值
        self.is_running = False
        self.plot_data = []
        self.fft_data = []
        self.fft_freqs = []
        
    def start_receiving(self):
        """開始接收數據的線程"""
        self.is_running = True
        self.receiver_thread = threading.Thread(target=self._receive_data)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()
    
    def stop_receiving(self):
        """停止接收數據"""
        self.is_running = False
        if hasattr(self, 'receiver_thread'):
            self.receiver_thread.join(timeout=1.0)
        self.serial_port.close()
    
    def _receive_data(self):
        """接收並解析數據的主循環"""
        buffer = bytearray()
        
        while self.is_running:
            if self.serial_port.in_waiting:
                # 讀取可用數據
                data = self.serial_port.read(self.serial_port.in_waiting)
                buffer.extend(data)
                
                # 處理完整數據包
                while len(buffer) >= PACKET_SIZE:
                    # 尋找數據包開頭
                    packet_start = -1
                    for i in range(len(buffer) - 1):
                        header_value = struct.unpack('<H', buffer[i:i+2])[0]
                        if header_value == HEADER_VALUE:
                            # 檢查是否有足夠數據構成完整包
                            if i + PACKET_SIZE <= len(buffer):
                                # 檢查 footer
                                footer_pos = i + PACKET_SIZE - 2
                                footer_value = struct.unpack('<H', buffer[footer_pos:footer_pos+2])[0]
                                if footer_value == FOOTER_VALUE:
                                    packet_start = i
                                    break
                    
                    if packet_start != -1:
                        # 提取並處理數據包
                        packet = buffer[packet_start:packet_start+PACKET_SIZE]
                        self._process_packet(packet)
                        # 移除已處理的數據
                        buffer = buffer[packet_start+PACKET_SIZE:]
                    else:
                        # 如果沒找到完整包，保留最後 PACKET_SIZE-1 字節
                        if len(buffer) > PACKET_SIZE:
                            buffer = buffer[-(PACKET_SIZE-1):]
                        break
            else:
                # 無數據時短暫休眠，避免 CPU 佔用過高
                time.sleep(0.001)
    
    def _process_packet(self, packet):
        """解析數據包並存儲 ADC 值"""
        # 解析 header
        header = struct.unpack('<H', packet[0:2])[0]
        
        # 解析 ADC 數據
        adc_data = []
        for i in range(ADC_BUFFER_SIZE):
            offset = 2 + i * 2
            value = struct.unpack('<H', packet[offset:offset+2])[0]
            adc_data.append(value)
        
        # 解析 footer
        footer = struct.unpack('<H', packet[-2:])[0]
        
        # 驗證 header 和 footer
        if header == HEADER_VALUE and footer == FOOTER_VALUE:
            # 將數據添加到緩衝區
            self.data_buffer.extend(adc_data)
            print(f"接收到 {len(adc_data)} 個數據點, 最新值: {adc_data[-1]}")
            self.plot_data = list(self.data_buffer)
            
            # 計算 FFT
            self._compute_fft()
        else:
            print(f"數據包錯誤! Header: {header:04x}, Footer: {footer:04x}")
    
    def _compute_fft(self):
        """計算數據的 FFT"""
        if len(self.plot_data) > 0:
            # 為FFT選擇2的次方個樣本點以優化速度 (最多不超過數據長度)
            n = min(1024, len(self.plot_data))
            
            # 取最新的n個點計算FFT
            data = np.array(self.plot_data[-n:])
            
            # 應用窗函數減少頻譜洩漏
            window = np.hamming(len(data))
            windowed_data = data * window
            
            # 計算 FFT
            fft_result = np.fft.rfft(windowed_data)
            
            # 計算幅度譜 (取絕對值並正規化)
            fft_magnitude = np.abs(fft_result) / len(data)
            
            # 只保留單側頻譜（因為信號是實數）
            self.fft_data = fft_magnitude
            
            # 計算頻率軸
            self.fft_freqs = np.fft.rfftfreq(len(data), d=1.0/SAMPLING_RATE)
    
    def plot_live_data(self, update_interval=100):
        """實時繪製接收到的 ADC 數據和 FFT"""
        plt.ion()  # 開啟互動模式
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 時域圖
        line1, = ax1.plot([], [])
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('ADC Value')
        ax1.set_title('Real-Time ADC Data')
        ax1.grid(True)
        
        # 頻域圖
        line2, = ax2.plot([], [])
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Real-Time FFT (100kHz Sampling Rate)')
        ax2.grid(True)
        
        # 調整子圖間距
        plt.tight_layout()
        
        while self.is_running:
            # 更新時域圖
            if self.plot_data:
                y_data = self.plot_data[-100:]  # 只顯示最新的100個點
                x_data = list(range(len(y_data)))
                
                line1.set_xdata(x_data)
                line1.set_ydata(y_data)
                
                ax1.set_xlim(0, len(y_data))
                ax1.set_ylim(min(y_data) - 100, max(y_data) + 100)
            
            # 更新頻域圖 (FFT)
            if self.fft_data.size > 0:
                # 只顯示有意義的頻率範圍 (可調整)
                max_freq_idx = min(len(self.fft_freqs), 
                                  np.searchsorted(self.fft_freqs, SAMPLING_RATE/2))
                
                line2.set_xdata(self.fft_freqs[:max_freq_idx])
                line2.set_ydata(self.fft_data[:max_freq_idx])
                
                ax2.set_xlim(0, self.fft_freqs[max_freq_idx-1])
                
                # 動態調整 y 軸以便更好地查看頻譜
                max_amplitude = max(self.fft_data[:max_freq_idx]) if len(self.fft_data) > 0 else 1
                ax2.set_ylim(0, max_amplitude * 1.1)
            
            # 更新圖表
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # 控制更新頻率
            plt.pause(update_interval/1000)
    
    def save_data_to_file(self, filename="adc_data.csv"):
        """將收集的數據保存到文件"""
        if self.plot_data:
            np.savetxt(filename, self.plot_data, delimiter=',')
            print(f"數據已保存到 {filename}")
        else:
            print("沒有數據可保存")

# 使用示例
if __name__ == "__main__":
    # 替換為你的 COM 端口
    PORT = "COM6"  # Windows 上通常是 COM 端口
    # PORT = "/dev/ttyUSB0"  # Linux 上通常是 /dev/ttyUSB0 或 /dev/ttyACM0
    
    receiver = ADCDataReceiver(PORT)
    try:
        receiver.start_receiving()
        # 開始繪製實時數據
        plotting_thread = threading.Thread(target=receiver.plot_live_data)
        plotting_thread.daemon = True
        plotting_thread.start()
        
        print("接收數據中... 按 Ctrl+C 停止")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n停止接收")
    finally:
        # 停止接收並保存數據
        receiver.stop_receiving()
        # receiver.save_data_to_file()