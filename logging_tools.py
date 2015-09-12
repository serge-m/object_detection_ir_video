__author__ = 's'
import logging
import logging.config
import json
import os

def setup_logging():
    """
    Setup logging module using 'logging_config.json' configuration file
    :return:
    """
    name_json = 'logging_config.json'
    path_json = os.path.join(os.path.dirname(__file__), name_json)
    with open(path_json, 'r') as f_json:
        dict_config = json.load(f_json)
    logging.config.dictConfig(dict_config)