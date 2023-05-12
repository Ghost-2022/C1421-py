#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Date: 2023/4/12 19:22
    @Auth: Jacob
    @Desc:
"""
import os
from pathlib import Path


class BaseConfig:
    BASE_DIR = Path(__file__).resolve().parent.parent

    IMG_DIR = os.path.join(BASE_DIR, 'static', 'img')
    FILE_DIR = os.path.join(BASE_DIR, 'static', 'file')
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class ProductionConfig(BaseConfig):
    DEBUG = False


class TestingConfig(BaseConfig):
    DEBUG = True

