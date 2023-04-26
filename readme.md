人脸识别系统

人脸检测+人脸对齐+人脸特征+近邻检索

1. 人脸检测
参考 https://github.com/hpc203/10kinds-light-face-detector-align-recognition，其中包含了10种轻量级人脸检测算法
   
2. 人脸特征提取
resnet算法，也可以使用facenet等算法
   
3. 近邻检索
Hnswlib库，也可以尝试使用fassi，annoy等，性能评估参考：https://zhuanlan.zhihu.com/p/37381294

4. yunnet+sfce
   
5. 相似度阈值，0.363（0.637）