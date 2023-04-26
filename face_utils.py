# author: sunshine
# datetime:2022/12/5 下午3:53
"""
人脸检测
人脸对齐
人脸特征
"""

import cv2
import numpy as np
from knn_util import HnswlibTool
from config import args

class FaceRecognitionOpenCV:
    def __init__(self, detect_model, feature_model, knn_path, input_size=(480, 640)):

        self.input_size = input_size
        self.detector = cv2.FaceDetectorYN.create(detect_model, config='', input_size=input_size,
                                                  score_threshold=args.score_threshold,
                                                  nms_threshold=args.nms_threshold)
        self.recognizer = cv2.FaceRecognizerSF.create(feature_model, config='')
        self.retriever = HnswlibTool(knn_path, dim=128, max_elements=args.max_elements)

    def face_embedding(self, img, threshold=0.363, retrieve=True):
        """

        :param img:
        :param retrieve:
        :return:
        """
        img = cv2.resize(img, self.input_size)
        # img_size = (img.shape[1], img.shape[0])
        # if img_size != self.input_size:
        #     self.detector.setInputSize(img_size)
        _, faces = self.detector.detect(img)
        if faces is None:
            raise Exception('未检测到人脸')

        features = []
        for face in faces:
            aligned_face = self.recognizer.alignCrop(img, face)
            feature = self.recognizer.feature(aligned_face)
            features.append(feature.squeeze())

        if retrieve:
            labels, distances = self.retriever.get_knn(np.stack(features))
            # output += ([l[0] for l in labels.tolist()], [1 - d[0] for d in distances.tolist()])
            # output += ([l[0]for l, d in zip(labels.tolist(), distances.tolist())])
            output = ([(l[0], 1 - d[0]) for l, d in zip(labels.tolist(), distances.tolist()) if 1 - d[0] > threshold])
            return features, output
        else:
            return features

    def put_face(self, img):
        """
        建库
        :return:
        """

        features = self.face_embedding(img, retrieve=False)
        if not features:
            raise Exception('请上传包含一张人脸的照片！')
        if len(features) > 1:
            raise Exception('请上传只有一张人脸的照片！')
        idx = self.retriever.add_item(np.stack(features))

        return idx

    def save(self):
        self.retriever.save()
