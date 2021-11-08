import os
import importlib

import hydra

def dynamic_import(import_path, alias=dict()):
    """dynamic import module and class

    :param str import_path: syntax 'module_name:class_name'
        e.g., 'AMINO.datamodule.datasets:TOYADMOS2_DATASET'
    :param dict alias: shortcut for registered class
    :return: imported class
    """
    if import_path not in alias and ":" not in import_path:
        raise ValueError(
            "import_path should be one of {} or "
            'include ":", e.g. "AMINO.datamodule.datasets:TOYADMOS2_DATASET" : '
            "{}".format(set(alias), import_path)
        )
    if ":" not in import_path:
        import_path = alias[import_path]

    module_name, objname = import_path.split(":")
    m = importlib.import_module(module_name)
    return getattr(m, objname)

def path_convert(path):
    path = path if os.path.isabs(path) else os.path.join(
        hydra.utils.get_original_cwd(), path
    )
    return path
