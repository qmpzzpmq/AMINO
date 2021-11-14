from AMINO.utils.dynamic_import import dynamic_import

def init_callbacks(callbacks_conf):
    callbacks = list()
    for callback in callbacks_conf:
        callback_class = dynamic_import(callback['select'])
        callbacks.append(callback_class(**callback['conf']))
    return callbacks
