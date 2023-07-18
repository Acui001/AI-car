import asyncio
import binascii

from bleak import BleakClient, BleakScanner

# 设备的Characteristic UUID，这是蓝牙设备的一种特性，具备通知属性(Notify)
par_notification_characteristic = "0000ffe1-0000-1000-8000-00805f9b34fb"

# 设备的Characteristic UUID，这是蓝牙设备的一种特性，具备写属性(Write)
par_write_characteristic = "0000ffe1-0000-1000-8000-00805f9b34fb"

# 设备的MAC地址，这是蓝牙设备的唯一标识
par_device_addr = "98:DA:20:05:35:A9"

# 标志变量，指示是否成功发送数据
data_sent = False

async def send_data():
    global data_sent
    async with BleakClient(par_device_addr) as client:
        # 将数据转换为十六进制
        data = b"A"
        hex_data = binascii.hexlify(data).decode()

        try:
            # 发送数据
            await client.write_gatt_char(par_write_characteristic, bytearray.fromhex(hex_data))
            data_sent = True
        except Exception as e:
            print(f"Failed to send data: {e}")
            data_sent = False


async def receive_data():
    async with BleakScanner() as scanner:
        devices = await scanner.discover()
        device = next((d for d in devices if d.address == par_device_addr), None)

        if device:
            async with BleakClient(device) as client:
                while True:
                    # 接收数据通知
                    notification = await client.read_gatt_char(par_notification_characteristic)
                    if notification:
                        # 解码并显示接收到的数据
                        received_data = binascii.unhexlify(notification.decode()).decode()
                        print(f"Received data: {received_data}")
                    else:
                        # 没有接收到数据
                        print("No data received.")

                    await asyncio.sleep(1)

# 创建事件循环
loop = asyncio.get_event_loop()

# 创建任务列表
tasks = [
    loop.create_task(send_data()),
    loop.create_task(receive_data())
]

# 执行任务并等待完成
try:
    loop.run_until_complete(asyncio.wait(tasks))
except KeyboardInterrupt:
    pass
finally:
    # 关闭事件循环
    loop.close()

# 显示数据发送情况
if data_sent:
    print("成功发送数据.")
else:
    print("Failed to send data.")
