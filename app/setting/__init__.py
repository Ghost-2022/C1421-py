#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Date: 2023/4/12 19:21
    @Auth: Jacob
    @Desc:
"""
from .config import DevelopmentConfig, ProductionConfig, TestingConfig

env_map = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig
}
