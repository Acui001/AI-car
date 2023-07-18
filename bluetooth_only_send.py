"""用于检测周边的蓝牙"""
import asyncio
from bleak import BleakScanner, BleakClient
# 需要注意的是这个UUID你可以在bule tooth.py中找到,但是你需要将前0000ffe2
# 其中
car_uuid = "0000ffe2-0000-1000-8000-00805f9b34fb"
data_to_send = bytearray([0x02])  # 要发送的字符数据
async def run():
    device_address = '98:DA:20:05:35:A9'  # 目标设备地址
    # 创建蓝牙解析器
    client = BleakClient(device_address)
    await client.connect()
    # 检查连接状态
    if client.is_connected:
        # 向设备写入数据
        result = await client.write_gatt_char(car_uuid, data_to_send, True)
        print(result)
        print("数据已发送到设备。")
    else:
        print("无法连接到设备。")
asyncio.run(run())


