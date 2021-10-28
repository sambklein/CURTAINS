import pathlib
import json
import os
import pandas as pd


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
    return str(pathlib.Path().absolute())


def register_experiment(sv_dir, exp_name, args):
    log_dict = vars(args)
    json_dict = json.dumps(log_dict)
    with open(sv_dir + '/images/' + exp_name + "_exp.json", "w") as file_name:
        json.dump(json_dict, file_name)


def read_experiment(sv_dir, exp_name, args):
    with open(sv_dir + exp_name + "_exp_info.json", "r") as file_name:
        json_dict = json.load(file_name)
    return json.loads(json_dict)


def make_slim(df, directory, filename, overwrite=False):
    slim_name = f'{directory}/slim_{filename}'
    if (not os.path.isfile(slim_name)) or overwrite:
        pd.to_hdf(slim_name, df.sample(10000))
