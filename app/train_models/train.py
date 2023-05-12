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
# 引入随机数
import random
# 引入画图的包
import matplotlib.pyplot as plt

train_core1 = {
    # 统一规范输入图片size大小
    "input_size": [3, 224, 224],
    # 图片分类问题需要分几类
    "class_dim": 20,
    # 设置学习率,可以设置0.001,0.005,0.01,0.02等进行调整
    "learning_rate":0.0002,
    # 使用GPU进行训练
    "use_gpu": True,
    # 第一次的训练轮数
    "num_epochs": 5, #训练轮数
    # 当达到期望需要的准确率就会立刻保存模型
    "last_acc":0.4
}

#按比例随机切割数据集 train : val = 9 : 1
train_ratio=0.9

train=open('train_split_list.txt','w')
val=open('val_split_list.txt','w')
with open('train_list.txt','r') as f:
    #with open('data_sets/cat_12/train_split_list.txt','w+') as train:
    lines=f.readlines()
    for line in lines:
        if random.uniform(0, 1) <= train_ratio:
            train.write(line)
        else:
            val.write(line)
train.close()
val.close()


## 获取植物叶片的数据
train_reader = paddle.batch(reader.train(), batch_size=32)
test_reader = paddle.batch(reader.val(), batch_size=32)

def resnet(input):
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
    return pool

## 定义输入层
image=fluid.layers.data(name='image',shape=train_core1["input_size"],dtype='float32')
label=fluid.layers.data(name='label',shape=[1],dtype='int64')
# print("lable={0}".format(label))


## 停止梯度下降
pool=resnet(image)
pool.stop_gradient=True

## 创建主程序来预训练
base_model_program=fluid.default_main_program().clone()
model=fluid.layers.fc(input=pool,size=train_core1["class_dim"],act='softmax')

## 定义损失函数和准确率函数
cost=fluid.layers.cross_entropy(input=model,label=label)
avg_cost=fluid.layers.mean(cost)
acc=fluid.layers.accuracy(input=model,label=label)


## 定义优化方法 这里使用Adam(减少调参的麻烦，实际上用SGD随机梯度下降也是可以的)
optimizer=fluid.optimizer.AdamOptimizer(learning_rate=train_core1["learning_rate"])
opts=optimizer.minimize(avg_cost)


## 定义训练场所——放在GPU训练
use_gpu=train_core1["use_gpu"]
place=fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe=fluid.Executor(place)

## 进行参数初始化
exe.run(fluid.default_startup_program())

# 每次重启后运行一次
## 预训练模型路径
src_pretrain_model_path='ResNet101_pretrained'

## 判断模型文件是否存在，存在ResNet101_pretrained文件夹中
def if_exit(var):
    path=os.path.join(src_pretrain_model_path,var.name)
    exist=os.path.exists(path)
    if exist:
      print('Load model: %s' % path)
      return exist

## 加载模型文件，且只加载存在模型的模型文件
fluid.io.load_vars(executor=exe,dirname=src_pretrain_model_path,predicate=if_exit,main_program=base_model_program)

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
    plt.savefig('train.png')
    plt.show()

##定义数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
# 保存模型的位置，存在models中
save_pretrain_model_path = 'models/step-1_model/'
last_acc = train_core1["last_acc"]
# 初始的准确率
now_acc = 0

for pass_id in range(train_core1["num_epochs"]):
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(), feed=feeder.feed(data),
                                        fetch_list=[avg_cost, acc])
        if batch_id % 50 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))
        if batch_id % 150 == 0:
            all_train_iter = all_train_iter + 32
            all_train_iters.append(all_train_iter)
            all_train_costs.append(train_cost[0])
            all_train_accs.append(train_acc[0])

    now_acc = train_acc
    if now_acc > last_acc and now_acc != 1:
        last_acc = now_acc
        ##删除旧的模型文件
        shutil.rmtree(save_pretrain_model_path, ignore_errors=True)
        ##创建保存模型文件记录
        os.makedirs(save_pretrain_model_path)
        ##保存参数模型
        fluid.io.save_params(executor=exe, dirname=save_pretrain_model_path)
draw_train_process("training", all_train_iters, all_train_costs, all_train_accs, "trainning cost", "trainning acc")