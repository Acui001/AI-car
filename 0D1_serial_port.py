import serial.tools.list_ports


# 扫描当前连接的串口设备
def scan_serial_ports():
    ports = serial.tools.list_ports.comports()
    available_ports = []

    for port in ports:
        available_ports.append(port.device)

    return available_ports


# 尝试与设备通信
def communicate_with_device(port_name):
    try:
        # 打开串口连接
        ser = serial.Serial(port_name, baudrate=9600, timeout=1)

        # 向设备发送数据
        ser.write(b'Hello, device!')

        # 从设备读取数据
        response = ser.readline()

        # 关闭串口连接
        ser.close()

        return response.decode().strip()

    except serial.SerialException as e:
        return str(e)


# 扫描并打印当前连接的串口设备
print("扫描串口中...")
ports = scan_serial_ports()
if ports:
    print("可用串口为:")
    for port in ports:
        print(port)
else:
    print("无可用串口.")

while True:
    # 选择要进行通信的设备
    if ports:
        selected_port = input("输入你想要通信的串口名: ")
        print("正在连接到设备...")
        response = communicate_with_device(selected_port)
        print("来自设备的回复:", response)
    else:
        print("指定串口不可用！")


