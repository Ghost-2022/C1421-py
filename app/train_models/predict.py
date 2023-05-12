import os
import json

import paddle
paddle.enable_static()
import paddle.fluid as fluid
import numpy as np
from PIL import Image
import sys

TOP_K = 1
DATA_DIM = 224

# 保存最终model的路径
SAVE_DIRNAME = './models/step_2_model'
abs_path = './leaf_train' # 测试文件夹的真实路径,这里为了方便起见,统一指向一个文件夹，但是test和train内的数据是不一样的
test_path = 'train_list.txt' # 记录测试图片对应类别的txt文件

use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(SAVE_DIRNAME, exe
                                                                                      # model_filename='model',
                                                                                      # params_filename='params'
                                                                                      # model_filename = 'fc_0.w_0',
                                                                                      # params_filename = 'params'
                                                                                      )


def real_infer_one_img(im):
    infer_result = exe.run(
        inference_program,
        feed={feed_target_names[0]: im},
        fetch_list=fetch_targets)

    # print(infer_result)
    # 打印预测结果
    mini_batch_result = np.argsort(infer_result)  # 找出可能性最大的列标，升序排列
    # print(mini_batch_result.shape)
    mini_batch_result = mini_batch_result[0][:, -TOP_K:]  # 把这些列标拿出来
    mini_batch_result = mini_batch_result.flatten()  # 拉平了，只吐出一个 array
    mini_batch_result = mini_batch_result[::-1]  # 逆序
    return mini_batch_result


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)
    return img


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


def process_image(img_path):
    img = Image.open(img_path)
    img = resize_short(img, target_size=256)
    img = crop_image(img, target_size=DATA_DIM, center=True)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img).astype(np.float32).transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std

    img = np.expand_dims(img, axis=0)
    return img


def convert_list(my_list):
    my_list = list(my_list)
    my_list = map(lambda x: str(x), my_list)
    # print('_'.join(my_list))
    return '_'.join(my_list)


def infer(file_path):
    im = process_image(file_path)
    result = real_infer_one_img(im)
    result = convert_list(result)
    return result


def get_test_classes(file_path):
    dict = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            dict["./" + parts[0]] = parts[1]
    return dict


# 字符串转数字
def str2int(s):
    try:
        return int(s)
    except:
        if ('-' == s[0]):
            return 0 - str2int(s[1:])
        elif s[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            num = 0
            for i in range(len(s)):
                if s[i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    num = num * 10 + int(s[i])
                else:
                    return num
        else:
            return 0


def createCSVFile(fruit_leaf_test_path):
    dict = get_test_classes(test_path)

    lines = []
    total_count = 0
    right_count = 0
    right_quantity = []
    all_quantity = []
    for i in range(20):
        right_quantity.append(0)
        all_quantity.append(0)
    # 获取所有的文件名
    img_paths = os.listdir(fruit_leaf_test_path)

    for file_name in img_paths:
        total_count += 1
        file_name = file_name
        file_abs_path = os.path.join(fruit_leaf_test_path, file_name).replace('\\',"/")
        result_classes = infer(file_abs_path)
        all_quantity[str2int(dict[file_abs_path])] = all_quantity[str2int(dict[file_abs_path])] + 1

        file_predict_classes = result_classes
        if file_predict_classes == dict[file_abs_path]:
            right_quantity[str2int(result_classes)] = right_quantity[str2int(result_classes)] + 1
            right_count += 1

        line = '%s,%s\n' % (file_name, file_predict_classes)
        lines.append(line)

    # print("每个类别实际对应数量为:{0}".format(right_quantity))
    # print("每个类别理应对应数量为:{0}".format(all_quantity))

    for i in range(20):
        all_quantity[i] = right_quantity[i] / all_quantity[i]

    # 获取一个预测结果的csv
    # with open('result.csv', 'w') as f:
    #     f.writelines(lines)
    print("最终测试得准确率为:{0}".format(right_count / total_count))
    print("每个类别测试对应准确率为:{0}".format(all_quantity))


createCSVFile(abs_path)
print("成功输出结果文件")