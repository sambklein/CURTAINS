import pathlib


def on_cluster():
    """
    :return: True if running job on cluster
    """
    p = pathlib.Path().absolute()
    id = p.parts[:3][-1]
    if id == 'beegfs':
        return True
    else:
        return False


def get_top_dir():
    p = pathlib.Path().absolute()
    id = p.parts[:3][-1]
    if id == 'samklein':
        sv_ims = '/Users/samklein/PycharmProjects/implicitBIBae'
    elif id == 'beegfs':
        sv_ims = '/srv/beegfs/scratch/groups/dpnc/atlas/BIB/implicitBIBae'
    else:
        raise ValueError('Unknown path for saving images {}'.format(p))
    return sv_ims
