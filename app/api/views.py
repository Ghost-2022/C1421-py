#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Date: 2023/4/15 10:32
    @Auth: Jacob
    @Desc:
"""
import os

from flask import request, current_app

from app.api import api
from app.train_models.vision import process_image, numbers1, numbers2, real_infer_one_img


@api.route('/train', methods=['POST'])
# @login_required
def get_train_result():
    """

    :return:
    """
    file = request.files.get('file')
    file_path = os.path.join(current_app.config['IMG_DIR'], file.filename)
    file.save(file_path)
    try:
        img = process_image(file_path)
        key = int(real_infer_one_img(img))
    except Exception as e:
        return {'code': '0001', 'message': e}
    return {'code': '0000', 'message': 'Success',
            'data': {'result': numbers1.get(key), 'suggestion': numbers2.get(key)}}
