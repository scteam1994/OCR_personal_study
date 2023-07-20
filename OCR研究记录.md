# OCR研究记录
## 参数设置
所有可修改预测参数在utility.py中的init_args中,在pip库paddleocr中未找到方便入口，可以在paddleocr.py中，手动
修改params.{参数}={value}

### use_dilation=True代表是否使用膨胀操作效果对比如下，上为默认，下为use_dilation=True
![default.png](asset/default.png)
![--use_dilation=True_0.6.png](asset/--use_dilation%3DTrue_0.6.png)
### det_db_box_thresh 代表检测框的阈值，只有置信度大于阈值的检测框才会被保留，效果对比如下，上为0.6，下为0.1

![default.png](asset/default.png)
![det_db_box_thresh_0.1.png](asset/det_db_box_thresh_0.1.png)
### 使用`--use_dilation=True`和`det_db_box_thresh=0.1`的效果如下
![--use_dilation=True_0.6.png](asset/--use_dilation%3DTrue_0.6.png)
![--use_dilation=True_0.1.png](asset/--use_dilation%3DTrue_0.1.png)
值得注意的是，使用pip库paddlocr 或paddleocr repo 中predict_system.py预测结果图中文字是否被检测出来不完全取决于det部分，在post_process方法中如果后续rec部分文字检测的置信度低于drop_score参数，那么图像中该位置的文字框会被舍弃掉。

### 如果想要直观的看到文字检测的效果，需要使用rec=False参数，这样就不会进行文字识别，而是直接将文字框画在图像上，效果如下
![det_db_box_thresh:0.1_result.jpg](asset/det_db_box_thresh%3A0.1_result.jpg)
![det_db_box_thresh:0.6_result.jpg](asset/det_db_box_thresh%3A0.6_result.jpg)
### det_db_thresh代表检测模型中DB算法二值化的阈值，具体效果为测试，但是预计修改后的会导致精读下降，建议不修改

### det_limit_side_len代表图像进入模型前压缩到多少像素，增大可显著提高分辨率图像的预测结果，但是会导致预测速度降低
## 优化方法
### 方向纠正
1.使用opencv 霍夫变化矫正

2.使用模型预测倾斜角度
### 文字检测
预计需要数据集1k张左右，标注所有预测框
### 文字识别
预计需要数据集中有50k左右个预测框，标注框所有文字
## 图像超分研究记录
 todo
# UIE研究记录

## UIE流程

## 训练结果可视化
在训练过程中，可以使用visualdl工具查看训练过程中的loss,f1变化情况，具体操作如下:
在命令行中输入`visualdl --logdir {logdir}` ,logdir中可方加入多个log文件，然后在浏览器中输入`http://localhost:8040/` 即可查看训练过程中的loss,f1变化情况

# 图像分类研究记录
拟采用tensorflow自写模型 

## 拉数据
root@192.168.3.210

root@192.168.3.210:/home/dataset/data-1/
## 模型结构
### 1.简单方案
使用纯卷积网络构建resnet主干的分类网络，使用全局池化代替全连接层，使用softmax代替sigmoid，使用交叉熵代替focal loss
### 2.复杂方案
~~在简单方案的基础上，加入图像ocr检测框作为辅助信息，做多模态学习~~

https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/ernie-lyout/README.zh.md
使用ocr全部结果和原始image作为辅助信息，做多模态学习,模型输入与UIE-X相同，但是模型结构不同，使用ernie-lyout模型，使用paddlepaddle框架
![2023-07-13 13-34-37屏幕截图.png](asset/2023-07-13%2013-34-37%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

#### 修改自定义数据集

[demo.py](asset/demo.py)
参照
#### paddlepaddle版本可能出现的问题
部分api名字随着版本的更新而改变，已知可能出现的问题如下：
paddlenlp/trainer/trainer.py line:1187，删除整个 if paddle.device.get_all_custom_device_type() is not None:下内容
paddlenlp/trainer/trainer.py line:1690，paddle.get_rng_state() ->paddle.get_cuda_rng_state()
#### 修改metrics
paddlenlp/trainer/layout_trainer.py line:66中修改metrics没有用，会在返回的时候丢弃所有self.metrics的内容.
因此需要直接修改metrics，metrics变量为字典类型，可以自定义metrics计算方法。
#### 第一次实验：
直接使用原始数据，各个类别的数据量如下：

![2023-07-14 13-12-12屏幕截图.png](asset/2023-07-14%2013-12-12%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

检测结果如下：
[metrics_1.txt](asset%2Fmetrics_1.txt)

#### 第二次实验：
策略调整，将数据量较多的类别的数据量减少到300，同时增加了更多指标，各个类别的metrics如下：

[metrics_2.txt](asset/metrics_2.txt)

发现other类无法分类，准备进行第三次实验删除other类


# 智慧文档（IE）记录

##使Taskflow("document_intelligence")返回文本对应box值

paddlenlp\taskflow\document_intelligence.py 中line:178开始加入：
```
 bboxes = []
                        for pred in preds:
                            start = pred["start"]
                            end = pred["end"]
                            boxes = example.ori_boxes[start:end + 1]
                            # combine boxes
                            if len(boxes) > 1:
                                box_x1 = min([boxes[i][0].left for i in range(len(boxes))])
                                box_y1 = min([boxes[i][0].top for i in range(len(boxes))])
                                box_x2 = max([boxes[i][0].right for i in range(len(boxes))])
                                box_y2 = max([boxes[i][0].bottom for i in range(len(boxes))])
                            bboxes.append([box_x1, box_y1, box_x2, box_y2])

                    all_predictions.append({"prompt": example_query, "result": preds,"bbox":bboxes})
```
