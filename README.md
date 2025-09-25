# PaddleObjectDetector
百度飞桨目标检测模型ios端侧SDK封装

原官方提供的demo地址如下： 

https://github.com/PaddlePaddle/Paddle-Lite-Demo

服务侧使用ppyoloe 训练出来的模型不可以直接在端侧上使用。 也不能将服务侧的模型使用opt工具转成.nb 给端侧使用。 （测试发现，服务侧模型可以转成.nb模型，但交给端侧代码使用时，不起作用）。 真正可以给端侧使用的模型是 picodet， 服务侧使用 picodet模型进行训练，训练完毕后，使用opt工具生成.nb 模型， 再将.nb模型用于端侧
