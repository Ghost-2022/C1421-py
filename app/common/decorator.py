#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Date: 2023/5/4 16:19
    @Auth: Jacob
    @Desc:
"""
import functools

from flask import g


def login_required(f):
    """让装饰器装饰的函数属性不会变 -- name属性'
    '第1种方法,使用functools模块的wraps装饰内部函数"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            if g.username == 1:
                return {'code': '0001', 'message': 'Token has expired'}, 401
            elif g.username == 2:
                return {'code': '0001', 'message': 'Token authentication failed'}, 401
            elif g.username == 2:
                return {'code': '0001', 'message': 'Illegal token'}, 401
            else:
                return f(*args, **kwargs)
        except BaseException as e:
            return {'code': '0001', 'message': 'Please login.'}, 401
    return wrapper
