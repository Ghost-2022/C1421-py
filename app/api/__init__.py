#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Date: 2023/4/12 18:57
    @Auth: Jacob
    @Desc:
"""
from flask import Blueprint

api = Blueprint('api', __name__, url_prefix='/api/',)


from . import views
