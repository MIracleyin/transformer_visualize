#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/6/1 15:41
# !@Author  : miracleyin @email: miracleyin@live.com
# !@file: utils.py

import torch

def display_tensorshape(is_display=True):
    # 只显示 tensor shape
    if is_display:
        old_repr = torch.Tensor.__repr__

        def tensor_info(tensor):
            return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(
                tensor)

        torch.Tensor.__repr__ = tensor_info