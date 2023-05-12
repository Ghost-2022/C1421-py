# 引入os库,方便路径操作
import os
# 引入文件操作模块
import shutil
# 引入百度paddle模块
import paddle as paddle
paddle.enable_static()
# 引入百度飞桨的fluid模块
import paddle.fluid as fluid
# 方便设置参数
from paddle.fluid.param_attr import ParamAttr
# 引入自写的reader库文件，里面定义了一些图像预处理的方法，例如裁剪和翻转等数据增强操作
import reader
# 引入日志库,方便记录操作的结果
import logging
# 引入画图的包
import matplotlib.pyplot as plt

## 获取数据
train_reader = paddle.batch(reader.train(), batch_size=32)
test_reader = paddle.batch(reader.val(), batch_size=32)

## 定义第二次训练的超参数
train_core2 = {
    # 统一输入大小
    "input_size": [3, 224, 224],
    # 分类数
    "class_dim": 20,
    # 定义lr学习率，与第一次相比我们减小了学习率
    "learning_rate":0.0001,
    # 定义sgd的学习率，与第一次相比改为SGD随机梯度下降训练
    "sgd_learning_rate":0.00001,
    # 设置lr学习率衰减(Learning rate decay)
    "lrepochs":[20, 40, 60, 80, 100],
    "lrdecay":[1, 0.5, 0.25, 0.1, 0.01, 0.002],
    # 使用GPU
    "use_gpu": True,
    # 设置第二次的训练轮数
    "num_epochs": 10,
    # 第二次期望的精确度，大于该期望将保存模型
    "last_acc":0.8

}

# 定义新的ResNet，加入fc层，对网络进行微调操作，实验证明将显著提升精度
def resnet(input, class_dim):
    def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1, act=None, name=None):
        conv = fluid.layers.conv2d(input=input,
                                   num_filters=num_filters,
                                   filter_size=filter_size,
                                   stride=stride,
                                   padding=(filter_size - 1) // 2,
                                   groups=groups,
                                   act=None,
                                   param_attr=ParamAttr(name=name + "_weights"),
                                   bias_attr=False,
                                   name=name + '.conv2d.output.1')
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(input=conv,
                                       act=act,
                                       name=bn_name + '.output.1',
                                       param_attr=ParamAttr(name=bn_name + '_scale'),
                                       bias_attr=ParamAttr(bn_name + '_offset'),
                                       moving_mean_name=bn_name + '_mean',
                                       moving_variance_name=bn_name + '_variance', )

    def shortcut(input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck_block(input, num_filters, stride, name):
        conv0 = conv_bn_layer(input=input,
                              num_filters=num_filters,
                              filter_size=1,
                              act='relu',
                              name=name + "_branch2a")
        conv1 = conv_bn_layer(input=conv0,
                              num_filters=num_filters,
                              filter_size=3,
                              stride=stride,
                              act='relu',
                              name=name + "_branch2b")
        conv2 = conv_bn_layer(input=conv1,
                              num_filters=num_filters * 4,
                              filter_size=1,
                              act=None,
                              name=name + "_branch2c")

        short = shortcut(input, num_filters * 4, stride, name=name + "_branch1")

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu', name=name + ".add.output.5")

    depth = [3, 4, 23, 3]
    num_filters = [64, 128, 256, 512]

    conv = conv_bn_layer(input=input, num_filters=64, filter_size=7, stride=2, act='relu', name="conv1")
    conv = fluid.layers.pool2d(input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

    for block in range(len(depth)):
        for i in range(depth[block]):
            if block == 2:
                if i == 0:
                    conv_name = "res" + str(block + 2) + "a"
                else:
                    conv_name = "res" + str(block + 2) + "b" + str(i)
            else:
                conv_name = "res" + str(block + 2) + chr(97 + i)
            conv = bottleneck_block(input=conv,
                                    num_filters=num_filters[block],
                                    stride=2 if i == 0 and block != 0 else 1,
                                    name=conv_name)

    pool = fluid.layers.pool2d(input=conv, pool_size=7, pool_type='avg', global_pooling=True)
    output = fluid.layers.fc(input=pool, size=class_dim, act='softmax')
    return output


# 定义新的ResNet，加入fc层，对网络进行微调操作，实验证明将显著提升精度
def resnet(input, class_dim):
    def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1, act=None, name=None):
        conv = fluid.layers.conv2d(input=input,
                                   num_filters=num_filters,
                                   filter_size=filter_size,
                                   stride=stride,
                                   padding=(filter_size - 1) // 2,
                                   groups=groups,
                                   act=None,
                                   param_attr=ParamAttr(name=name + "_weights"),
                                   bias_attr=False,
                                   name=name + '.conv2d.output.1')
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(input=conv,
                                       act=act,
                                       name=bn_name + '.output.1',
                                       param_attr=ParamAttr(name=bn_name + '_scale'),
                                       bias_attr=ParamAttr(bn_name + '_offset'),
                                       moving_mean_name=bn_name + '_mean',
                                       moving_variance_name=bn_name + '_variance', )

    def shortcut(input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck_block(input, num_filters, stride, name):
        conv0 = conv_bn_layer(input=input,
                              num_filters=num_filters,
                              filter_size=1,
                              act='relu',
                              name=name + "_branch2a")
        conv1 = conv_bn_layer(input=conv0,
                              num_filters=num_filters,
                              filter_size=3,
                              stride=stride,
                              act='relu',
                              name=name + "_branch2b")
        conv2 = conv_bn_layer(input=conv1,
                              num_filters=num_filters * 4,
                              filter_size=1,
                              act=None,
                              name=name + "_branch2c")

        short = shortcut(input, num_filters * 4, stride, name=name + "_branch1")

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu', name=name + ".add.output.5")

    depth = [3, 4, 23, 3]
    num_filters = [64, 128, 256, 512]

    conv = conv_bn_layer(input=input, num_filters=64, filter_size=7, stride=2, act='relu', name="conv1")
    conv = fluid.layers.pool2d(input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

    for block in range(len(depth)):
        for i in range(depth[block]):
            if block == 2:
                if i == 0:
                    conv_name = "res" + str(block + 2) + "a"
                else:
                    conv_name = "res" + str(block + 2) + "b" + str(i)
            else:
                conv_name = "res" + str(block + 2) + chr(97 + i)
            conv = bottleneck_block(input=conv,
                                    num_filters=num_filters[block],
                                    stride=2 if i == 0 and block != 0 else 1,
                                    name=conv_name)

    pool = fluid.layers.pool2d(input=conv, pool_size=7, pool_type='avg', global_pooling=True)
    output = fluid.layers.fc(input=pool, size=class_dim, act='softmax')
    return output

## 定义输入层
image = fluid.layers.data(name='image', shape=train_core2["input_size"], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
## 获取分类器
model = resnet(image,train_core2["class_dim"])

## 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

## 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

## 定义优化方法
optimizer=fluid.optimizer.SGD(learning_rate=train_core2["sgd_learning_rate"])
opts=optimizer.minimize(avg_cost)

## 定义一个使用GPU的执行器
use_gpu=train_core2["use_gpu"]
place=fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe=fluid.Executor(place)

## 进行参数初始化
exe.run(fluid.default_startup_program())

## 使用step-1处理后的的预训练模型
pretrained_model_path = 'models/step-1_model/'

## 加载经过处理的模型
fluid.io.load_params(executor=exe, dirname=pretrained_model_path)
last_acc=train_core2["last_acc"]

all_train_iter=0
all_train_iters=[]
all_train_costs=[]
all_train_accs=[]

def draw_train_process(title,iters,costs,accs,label_cost,lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs,color='red',label=label_cost)
    plt.plot(iters, accs,color='green',label=lable_acc)
    plt.legend()
    plt.grid()
    plt.savefig('train_lr.png')
    plt.show()
## 定义数据维度
feeder=fluid.DataFeeder(place=place,feed_list=[image,label])

now_acc=0
## 保存预测模型
save_path = './models/step_2_model/'
for pass_id in range(train_core2["num_epochs"]):
    ## 训练
    for batch_id,data in enumerate(train_reader()):
        train_cost,train_acc=exe.run(program=fluid.default_main_program(),feed=feeder.feed(data),fetch_list=[avg_cost,acc])
        if batch_id%50==0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))
        if batch_id%150==0:
            all_train_iter=all_train_iter+16
            all_train_iters.append(all_train_iter)
            all_train_costs.append(train_cost[0])
            all_train_accs.append(train_acc[0])
    ## 测试
    test_accs=[]
    test_costs=[]
    for batch_id,data in enumerate(test_reader()):
        test_cost,test_acc=exe.run(program=test_program,feed=feeder.feed(data), fetch_list=[avg_cost,acc])
        test_accs.append(test_acc[0])
        test_costs.append(test_cost[0])
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    now_acc=test_acc
    if now_acc>last_acc:
        last_acc=now_acc
        ## 删除旧的模型文件
        shutil.rmtree(save_path, ignore_errors=True)
        ## 创建保持模型文件目录
        os.makedirs(save_path)
        ## 保存预测模型
        #fluid.io.save_params(executor=exe, dirname=save_path)
        fluid.io.save_inference_model(save_path, feeded_var_names=[image.name], target_vars=[model], executor=exe)
draw_train_process("training", all_train_iters,all_train_costs,
                   all_train_accs,"trainning cost","trainning acc")

