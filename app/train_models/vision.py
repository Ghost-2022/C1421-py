# 展示图片

import cv2
import paddle

paddle.enable_static()
import paddle.fluid as fluid
import numpy as np
from PIL import Image

SAVE_DIRNAME = 'app/train_models/models/step_2_model'
TOP_K = 1
DATA_DIM = 224


def imshow(img_path):
    """
    :param img_path: 图片路径
    :return:
    """
    image = cv2.imread(img_path)
    print(type(image))
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imshow('image', image)
    cv2.waitKey(0)


place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
exe = fluid.Executor(place)
numbers1 = {0: '苹果褐斑病', 1: '苹果腐败', 2: '苹果锈病', 3: '苹果健康', 4: '蓝莓健康', 5: '樱桃健康',
            6: '樱桃白粉病', 7: '玉米灰斑病', 8: '玉米锈病', 9: '玉米健康', 10: '玉米枯叶病', 11: '葡萄锈病',
            12: '葡萄黑痘病', 13: '葡萄健康', 14: '葡萄褐斑病', 15: '柑橘黄龙病', 16: '桃子斑点病',
            17: '桃子健康', 18: '胡椒斑点病', 19: '胡椒健康'}
numbers2 = {3: '无', 4: '无', 5: '无', 9: '无', 13: '无', 17: '无', 19: '无',
            0: '（1）发病初期与积累期，交替使用内吸性治疗剂控制。43%戊唑醇悬浮剂3000倍液（戊唑醇套袋后使用，一年内最多用2次，否则对果实着色不良），'
               '也可选用40%氟硅唑（福星）乳油8000倍液，40%腈菌唑（信生）可湿性粉剂 8000倍液，62.25%腈菌唑+代森锰锌（仙生）可湿性粉剂600倍液等。'
               '（2）盛发期处理，除以上内吸性治疗剂外，还可在8-9月对已套袋的果园喷布1-2次波尔多液或多宁或必备，保护叶片，'
               '波尔多液的配比为 1（硫酸铜）:1.5-2（生石灰）:160-200（水）。',
            1: '将坏死组织彻底刮除、刮净，范围应控制到比变色死组织大出0.5-1厘米。'
               '树皮没有烂透的部位，只需要将上层病皮削除；病变深达木质部的部分，要刮到木质部并要连续刮治3-5年。'
               '5-9月份均可进行重刮皮法，以5-6月份最好。用锋利的刮子，对较粗的病干枝，刮去老翘皮、干死皮，将主干、主枝基部树皮表层刮去，要刮去1-2毫米表层活皮，露出白绿或黄白色皮层为止。',
            2: '常用的内吸性杀菌剂有氟硅唑，10%苯醚甲环唑水分散粒剂2500倍液，50%甲基硫菌灵可湿性粉剂600-800倍液，'
               '15%三唑酮可湿性粉剂2000倍液，43%戊唑醇水乳剂或悬浮剂4000倍液，25%丙环唑乳油2500倍液，12.5%腈菌唑水乳剂2000倍液等。',
            6: '(1)加强田间管理：合理密植、控制浇水、疏密枝条、避免偏施氮肥，使冠光透光。'
               '(2)落叶在落叶之前至萌芽前彻底清除，集中焚烧。早发现早摘果深埋，减少菌源。'
               '(3)药剂防治：南京博士邦农药杀菌剂醚菌酯、乙嘧酚、氟硅唑咪鲜胺防治桃树白粉病。',
            7: '发病初期喷洒75%百菌清可湿性粉剂500倍液或50%多菌灵可湿性粉剂600倍液、40%克瘟散乳油800～900倍液、'
               '50%苯菌灵可湿性粉剂1500倍液、25%苯菌灵乳油800倍液、20%三唑酮乳油1000倍液。',
            8: '用25%三唑酮可湿性粉剂1000～1 500倍液或12.5%速保利可湿性粉剂4 000倍液常量喷雾。在发病早期喷药控制传病中心，在病叶率达到6%时全田喷药防治.',
            10: '（1）喷洒75%百菌清可湿性粉剂600倍液或50%扑海因可湿性粉剂1000倍液、50%速克灵可湿性粉剂1500倍液、70%代森锰锌可湿性粉剂500倍液，隔7—15天1次，防治2～3次。'
                '(4)可选用10%世高1500倍液、或85%三氯异氰脲酸1500倍液、或80%乙蒜素1500倍液、或20%龙克菌600倍液等药剂，并可选择加配绿风95、十乐素、蓝色晶典、芸苔素内酯等营养调节剂混合使用。',
            11: '发病初期喷洒波美0.2~0.3度石硫合剂或45%晶体石硫合剂300倍液、20%三唑酮（粉锈宁）乳油1500~2000倍液、20%三唑酮·硫悬浮剂1500倍液、'
                '40%多·硫悬浮剂400~500倍液、20%百科乳剂2000倍液、25%敌力脱乳油3000倍液、25%敌力脱乳油4000倍液+15%三唑酮可湿性粉剂2000倍液、'
                '12.5%速保利可湿性粉剂4000~5000倍液，隔15~20天1次，防治1次或2次。',
            12: '该病是葡萄生产中的早期病害，喷药目标是防止幼嫩的叶、果、蔓梢发病。'
                '在搞好清园越冬防治的基础上，生长季节的关键用药时期是花前半月、落花70-80%和花后半月这3次。'
                '在开花前后各喷1次1:0.7:250的波尔多液或10%世高水分散粒剂2000-3000倍液或500-600倍的达科宁（百菌清）液或4%农抗120水剂400倍液'
                '或杜邦抑快净52.5%水分散粒剂2000-3000倍液，70%甲基托布津800-1000倍液或50%多菌灵600倍液。'
                '此后，每隔半月喷1次1:0.7:240的波尔多液或70%代森锰锌800倍液或可杀得2000（900-1200倍）或2000倍世高，可有效地控制葡萄黑痘病的发展。',
            14: '发病初期喷50%消菌灵可溶性粉剂1500倍液，或1:0.7:200倍波尔多液，或30%碱式硫酸铜悬浮剂400-500倍液，'
                '或70%代森锰锌可湿性粉剂500-600倍液，或75%百菌清可湿性粉剂600-700倍液，或50%甲基硫菌灵悬浮剂800倍液，'
                '或50%多菌灵可湿性粉剂700倍液。隔10-15天喷1次，连续防治3-4次。',
            15: '柑橘木虱是柑橘黄龙病唯一的田间传播媒介，是黄龙病近距离传播的主要途径。它在吸食黄龙病植株汁液后终身带毒，转到健康植株上危害时传染病菌。',
            16: '(1)加强桃园管理。桃园注意排水，增施有机肥，合理修剪，增强通透性。'
                '(2)药剂防治。落花后，喷洒70%代森锰锌可湿性粉剂500倍液或70%甲基硫菌灵超微可湿性粉剂1000倍液、75%百菌清可湿性粉剂700—800倍液、'
                '50%混杀硫悬浮剂500倍液，7—10天防治一次，共防3—4次。',
            18: '发病严重时，喷射50％多菌灵1000倍液。'
            }
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(SAVE_DIRNAME, exe)


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


# im = process_image("leaf_train/OH680.jpg")
# print(2352435234)
# print(numbers1.get(int(real_infer_one_img(im))))
# print(numbers2.get(int(real_infer_one_img(im))))
