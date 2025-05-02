import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from UIfile.gestureUI import Ui_gestureUI
from PyQt5.QtCore import QTimer, pyqtSignal, QMutex, QThread, QWaitCondition, QObject
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QIcon, QCloseEvent
from PyQt5.QtGui import QImage, QPixmap
import cv2
import myGesture

class myGesture_UI(QWidget, Ui_gestureUI):
    # 通过信号将识别结果和生成的命令发送出去（不直接控制无人机）
    signal_send_order_to_Tello = pyqtSignal(str, str)

    def __init__(self, tello=None):
        super(myGesture_UI, self).__init__()
        self.setupUi(self)

        # 如果未初始化 Tello 对象，给出警告（但后续仍走信号发送流程）
        if tello is None:
            print("警告: Tello对象未初始化")
        self.tello = tello

        self.dockWidget_HintHelp.setFloating(False)
        self.channel = 3  # BGR格式的图像有3个通道

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_and_get_order)

        self.cap = cv2.VideoCapture()
        self.pushButton_start_and_close.clicked.connect(self.start_or_stop)

        # 初始化手势识别模块（仅生成命令，不直接发送）
        self.myGesture = myGesture.myGesture(self.tello)

        # 取消跳帧，实时处理
        self.order_already_sent = ''
        self.move_distance_per_step = 20

        # 进度条仅显示实时状态
        self.progressBar.setMaximum(100)
        self.progressBar.setValue(100)
        self.label_is_ready.setText('加载完成')

        self.speed = 30  # 默认速度值

    def start_or_stop(self):
        """启动或关闭手势识别控制"""
        if not self.timer.isActive():
            self.cap.open(0)
            if not self.cap.isOpened():
                print("错误: 无法打开摄像头")
                return
            self.pushButton_start_and_close.setText('关闭手势控制')
            self.timer.start(33)  # 约30帧/秒（33ms间隔）
            print("手势控制已启动")
        else:
            self.timer.stop()
            self.cap.release()
            self.label_PC_Cam_frame_show.clear()
            self.pushButton_start_and_close.setText('开始手势控制')
            print("手势控制已关闭")

    def show_and_get_order(self):
        """读取视频帧，调用手势识别处理后显示处理结果，并通过信号发送识别结果和命令"""
        self.img_w = self.label_PC_Cam_frame_show.width()
        self.img_h = self.label_PC_Cam_frame_show.height()
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (self.img_w, self.img_h))
            # 调用手势识别模块，获取识别结果、生成的命令和处理后的图像
            result, order, debug_image = self.myGesture.getOrder(frame)

            # 显示处理后的画面（调试图像中包含手势特征点等标注）
            if debug_image is not None:
                processed_frame = cv2.resize(debug_image, (self.img_w, self.img_h))
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(processed_frame.data, self.img_w, self.img_h,
                              self.channel * self.img_w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.label_PC_Cam_frame_show.setPixmap(pixmap)

            # 如果有有效命令，则更新界面并通过信号发送识别结果和命令（不直接发送给无人机）
            if order:
                self.update_textBrower(result, order)
                self.signal_send_order_to_Tello.emit(result, order)

    def closeEvent(self, a0: QCloseEvent) -> None:
        """窗口关闭时释放资源"""
        self.dockWidget_HintHelp.close()
        self.timer.stop()
        self.cap.release()

    def get_move_distance_per_step(self, distance: int):
        """更新步长设置"""
        self.move_distance_per_step = distance

    def update_textBrower(self, result, order):
        """更新文本浏览器，显示已生成的命令信息"""
        if order in ('stop', 'takeoff', 'land'):
            self.order_already_sent = '\n>>> ' + result
        else:
            self.order_already_sent = '\n>>> ' + result + f' 速度:{self.speed}'
        self.textBrowser_order_already_sent.append(self.order_already_sent)

# 如果需要独立测试，可在此处添加主函数入口
if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    widget = myGesture_UI()
    widget.show()
    sys.exit(app.exec_())