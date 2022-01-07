import os
import sys
import logging

from AMINO.utils.dynamic_import import dynamic_import

def init_object(conf):
    class_obj = dynamic_import(conf['select'])
    real_obj = class_obj(**conf['conf'])
    return real_obj

def init_list_object(confs):
    out_list = []
    for conf in confs:
        if conf is None:
            continue
        out_list.append(init_object(conf))
    if len(out_list) > 0:
        return out_list
    return None