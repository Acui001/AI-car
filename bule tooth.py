"""
这是一个扫描蓝牙的mac地址和UUID地址的文件
作为一个入门文件
你可以在这里学习
"""
import asyncio
from bleak import BleakScanner
from bleak import BleakClient
import logging

logger = logging.getLogger(__name__)

async def find():
    # 创建 BleakScanner 对象，用于扫描周围的蓝牙设备
    # 你可以简单理解为一个蓝牙扫描解析器
    scanner = BleakScanner()
    # 只要有一个await操作都是异步
    # 在异步函数中遇到 await 语句时，它会暂停当前函数的执行，将控制权交回给事件循环（event loop），并等待被 await 的操作完成。
    # 在等待期间，事件循环可以继续执行其他任务，而不会被阻塞。
    # 异步函数中才能使用 await 关键字，因为它需要在异步上下文中执行。异步函数以 async 关键字定义，可以通过 async def 来定义异步函数
    # discover()这个函数的作用是持续扫描设备，持续时间为 timeout 秒，并返回发现的设备
    devices = await scanner.discover()
    print("寻找蓝牙中")
    for device in devices:
        # 注意体会这三个变量名的作用
        # 注意这三个变量名都与你的设备无关
        # 都是小车设备的东西
        uuids = device.metadata['uuids']  # 一种获取UUID的方法，但是需要注意虽然会有版本标红，但是不影响使用就不要改
        print(f"设备名: {device.name}, 设备地址: {device.address}, 设备UUIDs: {uuids}")
        print("找到一个蓝牙设备")
        if device.address == '98:DA:20:05:70:5D':  # 98:DA:20:05:70:5D这是一个设备的Mac码，我指定这个设备，我就可以直接去找出这个设备
            print("——————————————————————————————————————————————————")
            print(f"小车UUIDs为: {uuids}")
            print("——————————————————————————————————————————————————")
            print("发现小车设备")
        else:
            print("小车设备未找到")
asyncio.run(find())

