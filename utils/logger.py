import os
import time
import logging

def inial_logger(file):
    logger = logging.getLogger('log')
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(file)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def make_save_file(path):
    current_save_path = os.path.join(path, time.strftime("%m-%d %H:%M:%S", time.localtime()))
    if not os.path.exists(current_save_path):
        os.makedirs(current_save_path)
    return current_save_path


