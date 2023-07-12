from SerialClass import SerialAchieve  # 导入串口通讯类


class MainSerial:
    def __init__(self):
        # 定义串口变量
        self.port = None
        self.band = "115200"
        self.check = "无校验位"
        self.data = "8"
        self.stop = "1"
        print("波特率：" + self.band)
        self.myserial = SerialAchieve(int(self.band), self.check, self.data, self.stop)

    def button_OK_click(self):
        '''
        @ 串口打开函数
        :return:
        '''
        if self.port is None or self.port.isOpen() == False:
            self.myserial.open_port('COM6')
            print("打开串口成功")
        else:
            pass

    def button_Send_click(self, a):
        try:
            if self.myserial.port.isOpen():
                send_str1 = a
                if send_str1:
                    print("开始发送数据")
                    self.myserial.Write_data(send_str1)
                    print("发送数据成功")
                    print("发送数据：" + send_str1)
                else:
                    print("输入为空")
            else:
                print("串口没有打开")
        except:
            print("发送失败")


if __name__ == '__main__':
    my_ser1 = MainSerial()
    my_ser1.button_OK_click()
    my_ser1.button_Send_click(a="1232323")
    print()
