#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Date: 2023/4/15 13:40
    @Auth: Jacob
    @Desc:
"""
from flask import render_template, request, g

from app import create_app

app = create_app()


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    print(app.url_map)
    app.run(host='0.0.0.0')
