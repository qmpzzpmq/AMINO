from AMINO.utils.dynamic_import import dynamic_import

def init_loss(loss_conf):
    loss_class = dynamic_import(loss_conf['select'])
    loss = loss_class(**loss_conf['conf'])
    return loss