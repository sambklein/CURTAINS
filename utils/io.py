import pathlib


def on_cluster():
    """
    :return: True if running job on cluster
    """
    p = pathlib.Path().absolute()
    id = p.parts[:3][-1]
    if id == 'users':
        return True
    else:
        return False


def get_top_dir():
    p = pathlib.Path().absolute()
    id = p.parts[:3][-1]
    if id == 'samklein':
        sv_ims = '/Users/samklein/PycharmProjects/CURTAINS'
    elif id == 'users':
        sv_ims = '/home/users/s/senguptd/UniGe/Anomaly/curtains'
    else:
        raise ValueError('Unknown path for saving images {}'.format(p))
    return sv_ims
