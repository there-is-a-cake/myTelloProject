#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import cv2 as cv
import time
import csv

# 添加正确的导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import configargparse
from utils import CvFpsCalc
from gestures import GestureRecognition, GestureBuffer

# 添加项目根目录到导入路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def get_args():
    try:
        print('## Reading configuration ##')
        # 创建一个简单的对象来存储参数
        class Args:
            def __init__(self):
                self.device = 0
                self.width = 960
                self.height = 540
                self.use_static_image_mode = False
                self.min_detection_confidence = 0.7
                self.min_tracking_confidence = 0.5
                self.buffer_len = 5
        
        args = Args()
        print("参数设置完成!")
        return args
    except Exception as e:
        import traceback
        print(f"参数设置出错: {e}")
        traceback.print_exc()
        sys.exit(1)

def save_gesture_data(number, data, is_keypoint=True):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if is_keypoint:
        filename = f"model/keypoint_classifier/keypoint_user_{timestamp}.csv"
    else:
        filename = f"model/point_history_classifier/point_history_user_{timestamp}.csv"
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 追加模式打开文件
    with open(filename, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *data])
    
    print(f"成功保存数据到新文件: {filename}")
    return filename

def select_mode_interface():
    """显示模式选择界面并返回用户选择"""
    print("\n" + "="*50)
    print("手势识别系统 - 模式选择")
    print("="*50)
    print("请选择运行模式:")
    print("1. 普通模式 - 仅识别手势")
    print("2. 记录关键点模式 - 采集手部关键点数据")
    print("3. 记录历史点模式 - 采集手指轨迹数据")
    print("0. 退出程序")
    print("="*50)
    
    while True:
        try:
            choice = int(input("请输入选择 (0-3): "))
            if 0 <= choice <= 3:
                if choice == 1:
                    return 0  # 普通模式
                elif choice == 2:
                    return 1  # 关键点记录模式
                elif choice == 3:
                    return 2  # 历史点记录模式
                else:  # choice == 0
                    print("感谢使用，再见!")
                    sys.exit(0)
            else:
                print("无效选择，请输入0-3之间的数字")
        except ValueError:
            print("请输入有效的数字")

def select_gesture_number(mode):
    """在记录模式下选择手势编号"""
    if mode == 0:  # 普通模式不需要选择编号
        return -1
    
    print("\n" + "="*50)
    if mode == 1:
        print("记录关键点模式 - 选择手势编号")
    else:
        print("记录历史点模式 - 选择手势编号")
    print("="*50)
    print("请为要记录的手势选择一个编号 (0-10):")
    print("0-9: 不同的手势类别")
    print("="*50)
    
    while True:
        try:
            number = int(input("请输入手势编号 (0-10): "))
            if 0 <= number <= 10:
                return number
            else:
                print("无效选择，请输入0-10之间的数字")
        except ValueError:
            print("请输入有效的数字")

def main():
    # Argument parsing
    args = get_args()
    print("****10****")
    
    # 模式选择
    mode = select_mode_interface()
    number = -1
    
    # 如果是记录模式，选择手势编号
    if mode > 0:
        number = select_gesture_number(mode)
    
    print(f"\n已选择{'普通' if mode==0 else '记录关键点' if mode==1 else '记录历史点'}模式")
    if mode > 0:
        print(f"手势编号: {number}")
    
    # 初始化电脑摄像头
    print("\n正在初始化摄像头...")
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头。尝试其他摄像头索引...")
        for i in range(1, 3):
            cap = cv.VideoCapture(i)
            if cap.isOpened():
                print(f"成功打开摄像头 {i}")
                break
        if not cap.isOpened():
            print("无法打开任何摄像头，请检查设备连接或权限")
            return

    print("摄像头已成功打开")
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    print(f"摄像头分辨率: {width}x{height}")

    # 初始化手势识别和缓冲区
    print("正在初始化手势识别...")
    gesture_detector = GestureRecognition(args.use_static_image_mode, args.min_detection_confidence,
                                          args.min_tracking_confidence)
    gesture_buffer = GestureBuffer(buffer_len=args.buffer_len)
    print("手势识别初始化完成")

    # FPS 计算器
    cv_fps_calc = CvFpsCalc(buffer_len=10)
    
    print("\n开始手势识别...")
    print("按 'q' 退出程序，按 'k' 返回普通模式，按 'n' 进入记录关键点模式，按 'h' 进入记录历史点模式")
    print("在记录模式下，按数字键 (0-9) 可以选择手势编号")

    frame_count = 0
    while True:
        # 计算 FPS
        fps = cv_fps_calc.get()

        # 处理键盘输入
        key = cv.waitKey(1) & 0xff
        if key == ord('q'):  # 按 'q' 退出
            break
        elif key == ord('k'):  # 按 'k' 返回普通模式
            mode = 0
        elif key == ord('n'):  # 按 'n' 进入记录关键点模式
            mode = 1
            number = -1
        elif key == ord('h'):  # 按 'h' 进入记录历史点模式
            mode = 2
            number = -1
        if mode in [1, 2]:  # 在记录模式下
            if 48 <= key <= 57:  # 数字键 0-9
                number = key - 48
            elif key == 48 + 10:  # 处理数字 10（可以用其他按键，比如 'x'）
                number = 10

        # 从摄像头捕获图像
        ret, image = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        frame_count += 1
        if frame_count % 30 == 0:  # 每30帧打印一次信息
            print(f"读取了 {frame_count} 帧")

        # 识别手势
        debug_image, gesture_id = gesture_detector.recognize(image, number, mode)
        gesture_buffer.add_gesture(gesture_id)

        # 绘制调试信息
        debug_image = gesture_detector.draw_info(debug_image, fps, mode, number)

        # 显示图像
        cv.imshow('Gesture Recognition', debug_image)

    # 释放资源
    cap.release()
    cv.destroyAllWindows()
    print("程序已退出")

if __name__ == '__main__':
    main()