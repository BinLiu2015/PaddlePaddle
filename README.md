# PaddlePaddle
本项目基于PaddlePaddle的PaddleDetection套件开展。

为了提高复杂背景下苹果叶片病虫害的检测精度，在PPYOLOE+_s模型的基础上，分别在backbone增加了ESPBlock模块、在CSPResStage增加了自适应特征融合模块、并在backbone和head之间增加了跳跃的信息传递结构，使得AP@0.5:0.95增长了3.6%。
