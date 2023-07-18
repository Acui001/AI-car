"""用于检测周边的蓝牙"""
import asyncio
from bleak import BleakScanner, BleakClient
car_uuid = "0000ffe2-0000-1000-8000-00805f9b34fb"
speed = bytearray([0x01])  # 要发送的字符数据
left = bytearray([0x47])
async def send_data_to_device():
    device_address = '98:DA:20:05:35:A9'  # 目标设备地址
    client = BleakClient(device_address)
    await client.connect()
    await client.write_gatt_char(car_uuid, speed, True)
        # await client.write_gatt_char(car_uuid, left, True)
    print("数据已发送到设备。")


async def main():
    # 运行蓝牙数据发送函数
    await send_data_to_device()

# 在主函数中运行异步函数
asyncio.run(main())
