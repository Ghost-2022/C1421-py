#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Date: 2023/5/4 16:09
    @Auth: Jacob
    @Desc:
"""
from flask import g, request, Flask, current_app, jsonify
import jwt
from jwt import exceptions

import functools
import datetime


def create_token(username, password):
    # 构造payload
    payload = {
        'username': username,
        'password': password,  # 自定义用户ID
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)  # 超时时间
    }
    result = jwt.encode(payload=payload, key=current_app.config['SALT'], algorithm="HS256")
    return result


def verify_jwt(token, secret=None):
    """
    检验jwt
    :param token: jwt
    :param secret: 密钥
    :return: dict: payload
    """
    if not secret:
        secret = current_app.config['SALT']
    try:
        payload = jwt.decode(token, secret, algorithms=['HS256'])
        return payload
    except Exception:  # 'token已失效'
        return None
