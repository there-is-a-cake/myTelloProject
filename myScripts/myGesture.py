import os
import sys
import time

# 设置正确的导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
gestures_path = os.path.join(project_root, 'gestures')
utils_path = os.path.join(project_root, 'utils')

# 添加路径
for path in [project_root, gestures_path, utils_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

print("=== Python 路径信息 ===")
print(f"项目根目录：{project_root}")
print(f"gestures目录：{gestures_path}")
print(f"utils目录：{utils_path}")

# 导入必要的模块
print("\n=== 模块导入 ===")
try:
    from utils import CvFpsCalc
    print("✓ 成功导入 CvFpsCalc")
except Exception as e:
    print(f"× CvFpsCalc 导入失败：{str(e)}")

try:
    from gestures.gesture_recognition import GestureRecognition, GestureBuffer
    print("✓ 成功导入 GestureRecognition 和 GestureBuffer")
except Exception as e:
    print(f"× GestureRecognition 导入失败：{str(e)}")



class myGesture:
    def __init__(self, tello):
        """
        初始化手势识别和无人机控制（仅生成命令，不直接发送）。
        
        :param tello: Tello 无人机对象 (myTello 实例)，用于记录无人机状态，但不发送命令
        """
        self.tello = tello
        self.is_flying = False
        self.takeoff_in_progress = False  # 添加起飞进行中标志
        self.takeoff_start_time = 0       # 添加起飞开始时间
        self.takeoff_timeout = 5.0        # 起飞超时时间（秒）
        self.gesture_recognizer = GestureRecognition(
            use_static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.gesture_buffer = GestureBuffer(buffer_len=10)

        # RC 控制速度变量
        self.left_right_velocity = 0
        self.forw_back_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0

        # 命令控制
        self.last_command = None  # 上次发送的命令
        self.last_gesture_id = -1  # 上次识别到的手势ID
        self.last_command_time = 0  # 上次命令时间
        self.command_interval = 0.1  # 命令发送间隔（秒）
        print(f"myGesture 初始化: is_flying = {self.is_flying}")

    def getOrder(self, image):
        current_time = time.time()
        print(f"当前飞行状态: is_flying = {self.is_flying}")
        debug_image, gesture_id = self.gesture_recognizer.recognize(image)
        self.gesture_buffer.add_gesture(gesture_id)
        smoothed_gesture_id = self.gesture_buffer.get_gesture()

        # 检查起飞状态
        if self.takeoff_in_progress:
            if current_time - self.takeoff_start_time > self.takeoff_timeout:
                print("【警告】起飞超时")
                self.takeoff_in_progress = False
                self.is_flying = False
                return ("起飞超时", "stop", debug_image)
            elif current_time - self.takeoff_start_time > 2.0:  # 假设2秒后起飞完成
                self.confirm_takeoff_complete()
            return ("正在起飞...", "takeoff_in_progress", debug_image)

        # 检测手势变化并清零速度
        if smoothed_gesture_id != self.last_gesture_id:
            self._reset_velocities()  # 手势变化时清零所有速度
            print(f"【调试】手势变化，从 {self.last_gesture_id} 到 {smoothed_gesture_id}，速度已清零")

        # 计算命令
        command = self._update_velocities(smoothed_gesture_id)
        result = self._get_result_from_gesture(smoothed_gesture_id)
        
        if command == "takeoff":
            print("识别到手势2，开始起飞流程")
            self.takeoff_in_progress = True
            self.takeoff_start_time = current_time
        elif command == "up (continuous)" and self.is_flying:
            result = "向上"

        self.last_gesture_id = smoothed_gesture_id
        self.last_command = command

        if current_time - self.last_command_time >= self.command_interval:
            if self.is_flying and command:
                print(f"【调试】发送命令: {command}, 速度信息: 左右={self.left_right_velocity}, "
                    f"前后={self.forw_back_velocity}, 上下={self.up_down_velocity}, 旋转={self.yaw_velocity}")
            self.last_command_time = current_time

        return (result, command, debug_image)

    def _get_result_from_gesture(self, gesture_id):
        """返回手势对应的描述文本"""
        gesture_map = {
            0: "向前", 
            1: "停止", 
            2: "起飞" if not self.is_flying else "向上",  # 根据飞行状态返回不同描述
            3: "降落", 
            4: "向下",
            5: "向后", 
            6: "向左", 
            7: "向右", 
            8: "向左", 
            9: "顺时针旋转",  # 新增：顺时针旋转(CW)
            10: "逆时针旋转",  # 新增：逆时针旋转(CCW)
            -1: ""
        }
        return gesture_map.get(gesture_id, "")

    def _update_velocities(self, gesture_id):
        if self.takeoff_in_progress:
            return "takeoff_in_progress"
        
        self._reset_velocities()
        
        if gesture_id == 0:
            self.forw_back_velocity = 30
            return "forward (continuous)"
        elif gesture_id == 1:
            return "stop"
        elif gesture_id == 2:
            if not self.is_flying:
                self.takeoff_in_progress = True
                self.takeoff_start_time = time.time()
                return "takeoff"
            else:
                self.up_down_velocity = 30
                return "up (continuous)"
        elif gesture_id == 3:
            self.is_flying = False
            return "land"
        elif gesture_id == 4:
            self.up_down_velocity = -30
            return "down (continuous)"
        elif gesture_id == 5:
            self.forw_back_velocity = -30
            return "back (continuous)"
        elif gesture_id == 6:
            self.left_right_velocity = -30
            return "left (continuous)"
        elif gesture_id == 7:
            self.left_right_velocity = 30
            return "right (continuous)"
        elif gesture_id == 8:  # 另一个向左手势
            self.left_right_velocity = -30
            return "left (continuous)"
        elif gesture_id == 9:  # 顺时针旋转
            self.yaw_velocity = 30
            return "cw (continuous)"
        elif gesture_id == 10:  # 逆时针旋转
            self.yaw_velocity = -30
            return "ccw (continuous)"
        elif gesture_id == -1:  # 无手势
            return ""
        return ""

    def _reset_velocities(self):
        self.left_right_velocity = 0
        self.forw_back_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0

    def confirm_takeoff_complete(self):
        """确认起飞完成"""
        self.takeoff_in_progress = False
        self.is_flying = True
        print("【状态更新】起飞完成，可以接受其他手势命令")
