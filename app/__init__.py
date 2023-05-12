#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Date: 2023/4/12 18:57
    @Auth: Jacob
    @Desc:
"""
import os

from flask import Flask
from flask_cors import CORS
from flask_migrate import Migrate
import pymysql

from app.common.db import db
from app.setting import env_map
from app.api import api

pymysql.install_as_MySQLdb()

migrate = Migrate()


def create_app():
    app = Flask(__name__, template_folder='static/')
    env = os.getenv('APP_ENV', 'development')
    app.config.from_object(env_map.get(env))

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    app.register_blueprint(api, url_prefix='/api/')
    return app
