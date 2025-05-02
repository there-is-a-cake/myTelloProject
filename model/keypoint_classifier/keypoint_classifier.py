#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    def __init__(self, model_path='model/keypoint_classifier/keypoint_classifier.tflite'):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        # 应用特征工程 - 添加2个角度特征
        input_details_tensor_index = self.input_details[0]['index']
        
        # 转换为增强特征（与训练时相同的特征工程）
        enhanced_features = self._create_pinky_features(landmark_list)
        
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([enhanced_features], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))

        return result_index
    
    def _create_pinky_features(self, landmark_list):
        """为小拇指关键点创建特征工程"""
        # 将一维landmark_list转换回21个点的二维数组格式
        points = np.array(landmark_list).reshape(-1, 2)
        
        # 提取小拇指关键点(索引17-20)
        p1 = points[17]  # 小拇指根部
        p2 = points[18]  # 第二关节
        p3 = points[19]  # 第一关节
        p4 = points[20]  # 指尖
        
        # 计算三个关节的向量
        v1 = p2 - p1  # 基础关节向量
        v2 = p3 - p2  # 中间关节向量
        v3 = p4 - p3  # 指尖关节向量
        
        # 计算两个角度特征
        # 使用arctan2计算向量间夹角
        angle1 = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
        angle2 = np.arctan2(np.cross(v2, v3), np.dot(v2, v3))
        
        # 将角度特征添加到原始特征中
        return np.append(landmark_list, [angle1, angle2])
