import tellopy
import av
import cv2
import time
import numpy
import threading

class myTello(tellopy.Tello):
    def __init__(self):
        super(myTello, self).__init__()

        #建立连接
        self.connect()
        self.wait_for_connection(60.0)

        self.Tello_frame = ''  # 存储Tello传回来的图像
        
        # 添加重试机制
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                print(f"尝试初始化视频流 (第 {attempt+1}/{max_attempts} 次)")
                self.container = av.open(self.get_video_stream())  # 解码视频流的容器
                print("视频流连接成功")
                break
            except Exception as e:
                print(f"视频流初始化失败: {e}")
                if attempt == max_attempts - 1:
                    self.Tello_frame = 'decode error'
                    print("视频流初始化最终失败")
                time.sleep(1)  # 等待1秒再重试

        # 确保线程作为守护线程运行，这样主程序结束时它会自动终止
        self.video_process_thread = threading.Thread(target=self.__Video_process)
        self.video_process_thread.daemon = True  # 设置为守护线程

        self.vid_stream = self.container.streams.video[0]
        self.height = self.vid_stream.height
        self.width = self.vid_stream.width

        self.date_fmt = '%Y-%m-%d_%H%M%S'  # 日期格式
        
        # 添加状态标记
        self.decoder_ready = False
        
        self.video_process_thread.start()
        
        # 等待解码器就绪
        wait_time = 0
        while not self.decoder_ready and wait_time < 5:
            time.sleep(1)
            wait_time += 1
    
    def get_flight_data(self):
        """获取飞行数据"""
        try:
            if hasattr(self, 'lcdNumber_Height'):
                height = float(self.lcdNumber_Height.value())
                return {
                    'height': height,
                    'is_flying': height > 10  # 假设高度大于10cm认为在飞行
                }
        except:
            return {
                'height': 0,
                'is_flying': False
            }

    def send(self, message):
        """发送指令"""
        try:
            self.sock.sendto(message.encode(), self.tello_addr)
            print("你输入的指令是: " + message)
            return True
        except Exception as e:
            print("传输错误: " + str(e))
            return False

    def get_video_frame(self):
        '''返回当前的Tello图像'''
        return self.Tello_frame

    def __Video_process(self):
        try:
            '''单独线程，一直解码并处理来自无人机的视频流'''
            frame_skip = 300        #跳过前300帧
            while True:
                for frame in self.container.decode(video=0):
                    if 0 < frame_skip:
                        frame_skip = frame_skip - 1
                        continue
                    start_time = time.time()
                    # self.Tello_frame = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                    self.Tello_frame =numpy.array(frame.to_image())     #不需要色彩空间转换
                    cv2.waitKey(20)
                    if frame.time_base < 1.0 / 60:
                        time_base = 1.0 / 60
                    else:
                        time_base = frame.time_base
                    frame_skip = int((time.time() - start_time) / time_base)
        except Exception as e:
            print(f"视频解码错误: {e}")
            self.Tello_frame = 'decode error'
    def toggle_recording(self, output_path=None):
        """切换录制状态"""
        if not hasattr(self, 'record') or not self.record:
            # 开始录制
            self.record = True
            if output_path is None:
                output_path = f'Tello-Video-{time.strftime(self.date_fmt)}.mp4'
            
            self.out_file = av.open(output_path, 'w')
            self.out_stream = self.out_file.add_stream(
                'mpeg4', self.vid_stream.rate)
            self.out_stream.pix_fmt = 'yuv420p'
            self.out_stream.width = self.width
            self.out_stream.height = self.height
            print(f"开始录制视频到: {output_path}")
            return output_path
        else:
            # 停止录制
            self.record = False
            if hasattr(self, 'out_file'):
                self.out_file.close()
                print("视频录制已停止")

    def send_rc_control(self, left_right_velocity, forward_backward_velocity, 
                        up_down_velocity, yaw_velocity):
            """发送RC控制命令"""
            try:
                cmd = f'rc {left_right_velocity} {forward_backward_velocity} {up_down_velocity} {yaw_velocity}'
                return self.send(cmd)
            except Exception as e:
                print(f"RC控制命令发送失败: {e}")
                return False

    def send_control_command(self, command):
        """发送控制命令（起飞、降落等）"""
        try:
            response = self.send(command)
            print(f"控制命令 '{command}' 发送状态: {'成功' if response else '失败'}")
            return response
        except Exception as e:
            print(f"控制命令发送错误: {e}")
            return False

def main():
    """独立运行测试函数 - 直接显示Tello视频流"""
    print("正在连接Tello并启动视频流...")
    tello = myTello()
    
    try:
        print("视频流已启动，按'q'键退出")
        while True:
            frame = tello.get_video_frame()
            if isinstance(frame, str) and frame == 'decode error':
                print("视频解码错误，请检查连接")
                time.sleep(1)
                continue
                
            if frame is not None and len(frame) > 0:
                # 转换为BGR格式并显示
                cv2.imshow("Tello视频流", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("检测到Ctrl+C，正在关闭...")
    finally:
        # 确保正确关闭
        cv2.destroyAllWindows()
        tello.quit()
        print("已安全断开连接")

if __name__ == "__main__":
    main()
