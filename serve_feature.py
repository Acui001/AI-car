import asyncio
from bleak import BleakScanner, BleakClient

async def main():
    scanner = BleakScanner()
    devices = await scanner.discover()
    print("设备扫描中...")
    for device in devices:
        print(f"设备名: {device.name}, 设备地址Mac: {device.address}")
        async with BleakClient(device) as client:
            print("是否已连接: ", client.is_connected)
            for service in client.services:
                print(f"[服务Service] {service}")
                for char in service.characteristics:
                    print(f"\t[特征Characteristic] {char.uuid}")
                    if "read" in char.properties:
                        try:
                            value = await client.read_gatt_char(char.uuid)
                            print(f"\t\t[值Value] {value}")
                        except Exception as e:
                            print(f"\t\t读取特征值时发生错误: {e}")
                    else:
                        print("\t\t该特征不允许读取")
                    for descriptor in char.descriptors:
                        print(f"\t\t[描述符Descriptor] {descriptor.uuid}")

if __name__ == "__main__":
    asyncio.run(main())
