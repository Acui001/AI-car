"""用于检测周边的蓝牙"""
import asyncio
from bleak import BleakScanner, BleakClient


async def run():
    # 建立扫描器
    scanner = BleakScanner()
    devices = await scanner.discover()
    print("寻找蓝牙中")
    for device in devices:
        print(f"设备名: {device.name}, 设备地址: {device.address}, 设备UUIDs: {device.metadata['uuids']}")
        print("找到一个蓝牙设备")
        if device.address == '98:DA:20:05:35:A9':  # 98:DA:20:05:35:A9
            # uuids = device.metadata.get('uuids', [])
            # print(f"小车UUIDs为: {uuids}")
            print("发现小车设备")
            print("——————————————————————————————————————————————————")
            uuids = device.metadata['uuids']
            for uuid in uuids:
                print(f"小车UUID为: {uuid}")
            print("——————————————————————————————————————————————————")
            # 创建蓝牙解析器bluetooth_test_success_move.py
            car_uuid = "0000ffe2-0000-1000-8000-00805f9b34fb"
            data_to_send = bytearray([0x01])  # 要发送的字符数据
            # 创建蓝牙解析器
            client = BleakClient(device)
            await client.connect()
            while True:
                # 每秒向设备写入一次数据
                result = await client.write_gatt_char(car_uuid, data_to_send, True)
                print(result)
                print("数据已发送到设备。")
        else:
            print("小车设备未找到")

asyncio.run(run())


