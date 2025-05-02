import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import av
import numpy
import tellopy
import cv2
import os
import json
import math
import time
import sys
import traceback

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from AlphaPose.opt import opt

from AlphaPose.dataloader import ImageLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from AlphaPose.yolo.util import write_results, dynamic_write_results
from AlphaPose.SPPE.src.main_fast_inference import *

import os
import sys
from tqdm import tqdm
import time
from AlphaPose.fn import getTime

class AlphaPose_control:
    # 设置参数
    args = opt
    args.inputpath = "E:/myTelloProject-master/AlphaPose/duan_alphapose/photo/"
    args.outputpath = "E:/myTelloProject-master/AlphaPose/duan_alphapose/"
    args.sp = True
    args.dataset = 'coco'

    img_path = args.inputpath

    if not args.sp:
        torch.multiprocessing.set_start_method('forkserver', force=True)
        torch.multiprocessing.set_sharing_strategy('file_system')

    def __init__(self, tello=None):
        """初始化AlphaPose控制器
        
        Args:
            tello: Tello对象实例，用于控制无人机
        """
        self.tello = tello  # 改用tello而不是drone
        self.margin = 0.1   # 添加安全边界margin属性
        if tello is None:
            print("警告: 未传入Tello对象，部分功能可能无法使用")


    # 获取图像
    def GetImage(self, args):
        inputpath = args.inputpath
        inputlist = args.inputlist
        mode = args.mode

        if not os.path.exists(args.outputpath):
            os.mkdir(args.outputpath)

        for root, dirs, files in os.walk(inputpath):
            im_names = files
        return im_names

    # 下载模型
    def downModel(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load pose model
        print('Loading YOLO model...')
        pose_dataset = Mscoco()
        if self.args.fast_inference:
            pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            pose_model = InferenNet(4 * 1 + 1, pose_dataset)
        pose_model.to(device)
        pose_model.eval()

        return pose_model

    # 处理图像，提取关键点
    def Alphapose(self, im_names, pose_model):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load input images
        data_loader = ImageLoader(im_names, batchSize=self.args.detbatch, format='yolo').start()

        # Load detection loader
        sys.stdout.flush()
        det_loader = DetectionLoader(data_loader, batchSize=self.args.detbatch).start()
        det_processor = DetectionProcessor(det_loader).start()
        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }

        # Init data writer
        writer = DataWriter(self.args.save_video).start()

        data_len = data_loader.length()
        # Simplify tqdm usage to avoid handle errors
        im_names_desc = tqdm(range(data_len), desc="Processing images", disable=not self.args.profile)

        batchSize = self.args.posebatch
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                    continue
                ckpt_time, det_time = getTime(start_time)
                runtime_profile['dt'].append(det_time)
                # Pose Estimation
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)].to(device)
                    hm_j = pose_model(inps_j)
                    hm.append(hm_j)
                hm = torch.cat(hm)
                ckpt_time, pose_time = getTime(ckpt_time)
                runtime_profile['pt'].append(pose_time)
                hm = hm.cpu()
                writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])

                ckpt_time, post_time = getTime(ckpt_time)
                runtime_profile['pn'].append(post_time)

            if self.args.profile:
                # Update description only if profiling is enabled
                im_names_desc.set_description(
                    f'det time: {np.mean(runtime_profile["dt"]):.3f} | '
                    f'pose time: {np.mean(runtime_profile["pt"]):.2f} | '
                    f'post processing: {np.mean(runtime_profile["pn"]):.4f}'
                )
                sys.stdout.flush()  # Ensure output is flushed to avoid buffering issues

        print('Finish Model Running.')
        if (self.args.save_img or self.args.save_video) and not self.args.vis_fast:
            print('===========================> Rendering remaining images in the queue...')
            print(
                '===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
        while writer.running():
            pass
        writer.stop()
        final_result = writer.results()
        try:
            if final_result[0]['result']:
                return final_result[0]['result'][0]['keypoints']
            else:
                return None
        except:
            return None

    # 根据关键点分析动作，返回无人机命令
    def PoseFind(self, point_results):
        """根据关键点分析动作，返回无人机命令和可视化图像，同时控制活动范围"""
        # 提取关键点坐标
        LS = [int(point_results[5][0].item()), int(point_results[5][1].item())]  # 左肩
        RS = [int(point_results[6][0].item()), int(point_results[6][1].item())]  # 右肩
        LE = [int(point_results[7][0].item()), int(point_results[7][1].item())]  # 左肘
        RE = [int(point_results[8][0].item()), int(point_results[8][1].item())]  # 右肘
        LH = [int(point_results[9][0].item()), int(point_results[9][1].item())]  # 左手腕
        RH = [int(point_results[10][0].item()), int(point_results[10][1].item())] # 右手腕
        LHip = [int(point_results[11][0].item()), int(point_results[11][1].item())]  # 左髋
        RHip = [int(point_results[12][0].item()), int(point_results[12][1].item())]  # 右髋
        nose = [int(point_results[0][0].item()), int(point_results[0][1].item())]  # 鼻子

        # 加载图像并获取分辨率
        img = cv2.imread(os.path.join(self.img_path, 'frame.jpg'))
        if img is None:
            print("Error: Could not load image.")
            return "stop", None
        image_width, image_height = img.shape[1], img.shape[0]

        # 计算臂展长度 - 左臂和右臂
        left_arm_length = math.sqrt((LS[0] - LE[0])**2 + (LS[1] - LE[1])**2) + \
                        math.sqrt((LE[0] - LH[0])**2 + (LE[1] - LH[1])**2)
        right_arm_length = math.sqrt((RS[0] - RE[0])**2 + (RS[1] - RE[1])**2) + \
                        math.sqrt((RE[0] - RH[0])**2 + (RE[1] - RH[1])**2)
        
        # 取较长的臂展长度作为安全距离参考
        arm_span = max(left_arm_length, right_arm_length)
        
        # 计算肩宽用于姿势识别阈值
        shoulder_width = math.sqrt((LS[0] - RS[0])**2 + (LS[1] - RS[1])**2)
        
        # 定义多级边界 - 基于臂展长度
        # 危险边界 - 超出则停止控制
        danger_margin_x = min(arm_span * 1.0 / image_width, 0.2)  # 最大不超过20%
        danger_margin_y = min(arm_span * 1.0 / image_height, 0.2)  # 最大不超过20%
        
        # 警告边界 - 进入则给出警告
        warning_margin_x = min(arm_span * 1.5 / image_width, 0.3)  # 最大不超过30%
        warning_margin_y = min(arm_span * 1.5 / image_height, 0.3)  # 最大不超过30%
        
        # 定义边界像素位置
        danger_x_min = image_width * danger_margin_x
        danger_x_max = image_width * (1 - danger_margin_x)
        danger_y_min = image_height * danger_margin_y
        danger_y_max = image_height * (1 - danger_margin_y)
        
        warning_x_min = image_width * warning_margin_x
        warning_x_max = image_width * (1 - warning_margin_x)
        warning_y_min = image_height * warning_margin_y
        warning_y_max = image_height * (1 - warning_margin_y)
        
        # 边界状态标志
        in_danger_zone = False
        in_warning_zone = False
        danger_direction = []  # 记录危险方向
        
        # 计算身体中心位置
        center_x = (LS[0] + RS[0]) / 2
        center_y = (LS[1] + RS[1]) / 2
        
        # 检查关键点是否超出安全范围
        key_points = [LS, RS, LE, RE, LH, RH, LHip, RHip, nose]
        for point in key_points:
            if (point[0] < danger_x_min or point[0] > danger_x_max or 
                point[1] < danger_y_min or point[1] > danger_y_max):
                in_danger_zone = True
                
                # 确定危险方向
                if point[0] < danger_x_min:
                    if "左" not in danger_direction:
                        danger_direction.append("左")
                elif point[0] > danger_x_max:
                    if "右" not in danger_direction:
                        danger_direction.append("右")
                
                if point[1] < danger_y_min:
                    if "上" not in danger_direction:
                        danger_direction.append("上")
                elif point[1] > danger_y_max:
                    if "下" not in danger_direction:
                        danger_direction.append("下")
                
            elif (point[0] < warning_x_min or point[0] > warning_x_max or 
                point[1] < warning_y_min or point[1] > warning_y_max):
                in_warning_zone = True
        
        # 控制无人机灯光 - 根据边界状态
        if hasattr(self, 'tello') and self.tello is not None:
            current_time = int(time.time())
            
            if in_danger_zone:
                # 危险区域 - 红色快速闪烁
                if current_time % 2 == 0:
                    self.tello.send_command("EXT led 255 0 0")  # 红色
                else:
                    self.tello.send_command("EXT led 0 0 0")    # 黑色(关闭)
                    
            elif in_warning_zone:
                # 警告区域 - 根据方向使用不同颜色
                # 检查接近哪个方向的边界
                if center_x < warning_x_min * 1.3:  # 接近左边界
                    if current_time % 3 == 0:
                        self.tello.send_command("EXT led 0 0 255")  # 蓝色表示左边界
                    else:
                        self.tello.send_command("EXT led 0 0 0")
                elif center_x > warning_x_max * 0.7:  # 接近右边界
                    if current_time % 3 == 0:
                        self.tello.send_command("EXT led 255 165 0")  # 橙色表示右边界
                    else:
                        self.tello.send_command("EXT led 0 0 0")
                elif center_y < warning_y_min * 1.3:  # 接近上边界
                    if current_time % 3 == 0:
                        self.tello.send_command("EXT led 255 255 0")  # 黄色表示上边界
                    else:
                        self.tello.send_command("EXT led 0 0 0")
                elif center_y > warning_y_max * 0.7:  # 接近下边界
                    if current_time % 3 == 0:
                        self.tello.send_command("EXT led 128 0 128")  # 紫色表示下边界
                    else:
                        self.tello.send_command("EXT led 0 0 0")
                else:
                    # 一般警告 - 黄色慢闪
                    if current_time % 4 == 0:
                        self.tello.send_command("EXT led 255 255 0")  # 黄色
                    else:
                        self.tello.send_command("EXT led 0 0 0")
            else:
                # 安全区域 - 绿色常亮
                self.tello.send_command("EXT led 0 255 0")  # 绿色
        
        # 如果在危险区域，立即停止并给出明确提示
        if in_danger_zone:
            print(f'---------------------------人体超出安全范围 {",".join(danger_direction)}边界 - 停止移动--------------------------')
            cv2.putText(img, f"警告: 已超出{','.join(danger_direction)}边界", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, "请做出相反方向动作回到安全区域", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 红色边框闪烁警告
            thickness = 5
            current_time = int(time.time() * 2)  # 2Hz闪烁
            if current_time % 2 == 0:
                cv2.rectangle(img, (0, 0), (image_width, image_height), (0, 0, 255), thickness)
            
            return "stop", img

        # 根据人体位置确定允许的操作方向
        allowed_directions = {
            "up": True, "down": True,
            "left": True, "right": True,
            "forward": True, "back": True,
            "stop": True, "land": True
        }
        
        # 如果接近边界，限制相应方向的移动
        if center_x < warning_x_min * 1.3:  # 接近左边界
            allowed_directions["left"] = False
        if center_x > warning_x_max * 0.7:  # 接近右边界
            allowed_directions["right"] = False
        if center_y < warning_y_min * 1.3:  # 接近上边界
            allowed_directions["up"] = False
        if center_y > warning_y_max * 0.7:  # 接近下边界
            allowed_directions["down"] = False

        # 姿势识别逻辑
        len_threshold = shoulder_width * 0.5
        left_arm_angle = calc_angle(LS, LE, LH)
        right_arm_angle = calc_angle(RS, RE, RH)
        command = "stop"

        # 识别姿势命令
        if (LS[1] - LE[1] >= len_threshold and LE[1] - LH[1] >= len_threshold and
            RS[1] - RE[1] >= len_threshold and RE[1] - RH[1] >= len_threshold):
            command = "up"
            print('---------------------------双手高举 - 上升--------------------------')
        elif (LH[1] - LE[1] >= len_threshold and LE[1] - LS[1] >= len_threshold and
            RH[1] - RE[1] >= len_threshold and RE[1] - RS[1] >= len_threshold):
            command = "stop"
            print('---------------------------双手下垂 - 悬停--------------------------')
        elif (left_arm_angle > 150 and right_arm_angle > 150 and
            abs(LH[1] - LE[1]) <= len_threshold and abs(LE[1] - LS[1]) <= len_threshold and
            abs(RH[1] - RE[1]) <= len_threshold and abs(RE[1] - RS[1]) <= len_threshold):
            command = "forward"
            print('---------------------------双手伸直 - 前进--------------------------')
        elif (LH[1] > LS[1] and LH[1] - LS[1] >= len_threshold and abs(LH[1] - LHip[1]) <= len_threshold and
            RH[1] > RS[1] and RH[1] - RS[1] >= len_threshold and abs(RH[1] - RHip[1]) <= len_threshold):
            command = "back"
            print('---------------------------双手叉腰 - 后退--------------------------')
        elif (left_arm_angle > 150 and abs(LE[1] - LS[1]) <= len_threshold and
            RH[1] - RE[1] >= len_threshold and RE[1] - RS[1] >= len_threshold):
            command = "right"
            print('---------------------------左平举右垂 - 向左--------------------------')
        elif (LH[1] - LE[1] >= len_threshold and LE[1] - LS[1] >= len_threshold and
            right_arm_angle > 150 and abs(RE[1] - RS[1]) <= len_threshold):
            command = "left"
            print('---------------------------左垂右平举 - 向右--------------------------')
        elif (LS[1] - LE[1] >= len_threshold and LE[1] - LH[1] >= len_threshold and
            RH[1] - RE[1] >= len_threshold and RE[1] - RS[1] >= len_threshold):
            command = "land"
            print('---------------------------左高举右下垂 - 降落--------------------------')
        elif (LH[1] - LE[1] >= len_threshold and LE[1] - LS[1] >= len_threshold and
            RS[1] - RE[1] >= len_threshold and RE[1] - RH[1] >= len_threshold):
            command = "down"
            print('---------------------------左下垂右高举 - 下降--------------------------')

        # 检查命令是否被允许
        if not allowed_directions.get(command, True):
            original_command = command
            command = "stop"
            print(f"---------------------------命令 {original_command} 被安全边界限制为停止--------------------------")
            
            # 如果命令被限制，使灯光闪烁红色提示
            if hasattr(self, 'tello') and self.tello is not None:
                self.tello.send_command("EXT led 255 0 0")  # 红色灯光提示被限制
                time.sleep(0.2)
                self.tello.send_command("EXT led 0 0 0")    # 熄灭
                time.sleep(0.2)
                self.tello.send_command("EXT led 255 0 0")  # 再次闪烁

        # 可视化关键点和连线
        point_list = [
            (nose[0], nose[1]), (int(point_results[1][0].item()), int(point_results[1][1].item())),
            (int(point_results[2][0].item()), int(point_results[2][1].item())),
            (int(point_results[3][0].item()), int(point_results[3][1].item())),
            (int(point_results[4][0].item()), int(point_results[4][1].item())),
            (LS[0], LS[1]), (RS[0], RS[1]), (LE[0], LE[1]), (RE[0], RE[1]),
            (LH[0], LH[1]), (RH[0], RH[1]), (LHip[0], LHip[1]), (RHip[0], RHip[1]),
            (int(point_results[13][0].item()), int(point_results[13][1].item())),
            (int(point_results[14][0].item()), int(point_results[14][1].item())),
            (int(point_results[15][0].item()), int(point_results[15][1].item())),
            (int(point_results[16][0].item()), int(point_results[16][1].item()))
        ]
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4), (9, 7), (7, 5), (5, 6), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        for point in point_list:
            cv2.circle(img, point, 1, (0, 0, 255), 4)
        for start, end in connections:
            cv2.line(img, point_list[start], point_list[end], (0, 255, 0), 1, 4)

        # 可视化命令信息
        cv2.putText(img, f"Command: {command}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # 绘制危险区域边界
        cv2.rectangle(img, (int(danger_x_min), int(danger_y_min)), 
                    (int(danger_x_max), int(danger_y_max)), (0, 0, 255), 2)
        
        # 绘制警告区域边界
        cv2.rectangle(img, (int(warning_x_min), int(warning_y_min)), 
                    (int(warning_x_max), int(warning_y_max)), (0, 255, 255), 1)
        
        # 在警告区域内闪烁警告
        if in_warning_zone:
            current_time = int(time.time() * 2)
            if current_time % 2 == 0:
                if center_x < warning_x_min * 1.3:
                    cv2.putText(img, "← 接近左边界", (int(center_x), int(center_y) - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(img, "无人机灯光: 蓝色闪烁", (10, image_height - 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                if center_x > warning_x_max * 0.7:
                    cv2.putText(img, "接近右边界 →", (int(center_x) - 120, int(center_y) - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(img, "无人机灯光: 橙色闪烁", (10, image_height - 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                if center_y < warning_y_min * 1.3:
                    cv2.putText(img, "接近上边界", (int(center_x) - 60, int(center_y) - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(img, "无人机灯光: 黄色闪烁", (10, image_height - 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if center_y > warning_y_max * 0.7:
                    cv2.putText(img, "接近下边界", (int(center_x) - 60, int(center_y) + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(img, "无人机灯光: 紫色闪烁", (10, image_height - 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)
        else:
            cv2.putText(img, "无人机灯光: 绿色常亮", (10, image_height - 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示臂展信息
        cv2.putText(img, f"臂展长度: {arm_span:.1f}px", (10, image_height - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 显示受限方向
        restricted = [dir for dir, allowed in allowed_directions.items() if not allowed and dir in ["up", "down", "left", "right"]]
        if restricted:
            restricted_cn = []
            for r in restricted:
                if r == "up": restricted_cn.append("上升")
                elif r == "down": restricted_cn.append("下降")
                elif r == "left": restricted_cn.append("向左")
                elif r == "right": restricted_cn.append("向右")
            cv2.putText(img, f"受限方向: {','.join(restricted_cn)}", (10, image_height - 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        return command, img
    # 删除文件函数，用于清空文件夹内图像
    def del_files(self, path_file):
        ls = os.listdir(path_file)
        for i in ls:
            f_path = os.path.join(path_file, i)
            # 判断是否是一个目录,若是,则递归删除
            if os.path.isdir(f_path):
                self.del_files(f_path)
            else:
                os.remove(f_path)


def calc_angle(p1, p2, p3):
        """计算三个点之间的夹角（以 p2 为顶点），返回角度（度）"""
        v1 = [p1[0] - p2[0], p1[1] - p2[1]]
        v2 = [p3[0] - p2[0], p3[1] - p2[1]]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if mag1 * mag2 == 0:  # 避免除以零
            return 0
        angle = math.degrees(math.acos(min(1.0, max(-1.0, dot / (mag1 * mag2)))))
        return angle
