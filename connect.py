import asyncio
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
# FF074064-5596-EA11-8104-BCE92F672061
par_notification_characteristic = "0000ffe1-0000-1000-8000-00805f9b34fb"
par_device_addr = "98:DA:20:05:35:A9"#98:DA:20:05:35:A9
disconnected_event = asyncio.Event()
# 定义断开连接的回调函数，这个函数会在设备断开连接时被调用
def disconnected_callback(client):
    print("Disconnected callback called!")
    disconnected_event.set()
async def connect_device():
    print("starting scan...")
    device = await BleakScanner.find_device_by_address(
        par_device_addr, cb=dict(use_bdaddr=False)  # use_bdaddr判断是否是MOC系统
    )
    if device is None:
        print("could not find device with address :", par_device_addr)
        return None

    print("connecting to device...")
    client = BleakClient(device, disconnected_callback=disconnected_callback)
    await client.connect()
    print("Connected")

    return client
# 运行连接部分
client = asyncio.run(connect_device())
# 设备的Characteristic UUID，这是蓝牙设备的一种特性，具备写属性(Write)
par_write_characteristic = "0000ffe2-0000-1000-8000-00805f9b34fb"
send_str = bytearray([0x41])   # 舵机打直
send_right = bytearray([0x42])  # 舵机右转
send_left = bytearray([0x47])  # 舵机左转
speed_up = bytearray([0x58])   # 电机加速100
speed_down = bytearray([0x59])  # 电机减速100
# 定义一个回调函数，当设备发送通知时会被调用，这里只是简单地打印出接收到的数据
def notification_handler(characteristic: BleakGATTCharacteristic, data: bytearray):
    print("rev data:", data)
async def communicate_device(client):
    # 开始接收设备的通知，当设备发送通知时，会调用notification_handler函数

    await client.start_notify(par_notification_characteristic, notification_handler)

    await client.write_gatt_char(par_write_characteristic, speed_up, True)
    result = await client.write_gatt_char(par_write_characteristic, send_right, True)
    print(result)
    # await asyncio.sleep(1)  # 每隔1秒发送一次数据

# 运行通信部分
if client is not None:
    asyncio.run(communicate_device(client))

