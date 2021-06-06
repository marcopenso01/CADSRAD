import os
import numpy as np
import logging
import cv2
import shutil
import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import tensorflow as tf

import configuration as config
import read_data

# Set SGE_GPU environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

log_dir = os.path.join(config.log_root, config.experiment_name)


def get_latest_model_checkpoint_path(folder, name):
    '''
    Returns the checkpoint with the highest iteration number with a given name
    :param folder: Folder where the checkpoints are saved
    :param name: Name under which you saved the model
    :return: The path to the checkpoint with the latest iteration
    '''

    iteration_nums = []
    for file in glob.glob(os.path.join(folder, '%s*.meta' % name)):

        file = file.split('/')[-1]
        file_base, postfix_and_number, rest = file.split('.')[0:3]
        it_num = int(postfix_and_number.split('-')[-1])

        iteration_nums.append(it_num)

    latest_iteration = np.max(iteration_nums)

    return os.path.join(folder, name + '-' + str(latest_iteration))


def run_training(continue_run):
        
        logging.info('EXPERIMENT NAME: %s' % config.experiment_name)
        
        init_step = 0
        
        if continue_run:
        logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!! Continuing previous run !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        try:
            init_checkpoint_path = get_latest_model_checkpoint_path(log_dir, 'model.ckpt')
            logging.info('Checkpoint path: %s' % init_checkpoint_path)
            init_step = int(init_checkpoint_path.split('/')[-1].split('-')[-1]) + 1  # plus 1 b/c otherwise starts with eval
            logging.info('Latest step was: %d' % init_step)
        except:
            logging.warning('!!! Didnt find init checkpoint. Maybe first run failed. Disabling continue mode...')
            continue_run = False
            init_step = 0
        
        # load data
        data = read_data.load_and_maybe_process_data(
                input_folder=config.data_root,
                preprocessing_folder=config.preprocessing_folder,
                mode=config.data_mode,
                size=config.image_size,
                target_resolution=config.target_resolution,
                force_overwrite=False
            )


def main():

    continue_run = True
    if not tf.io.gfile.exists(log_dir):
        tf.io.gfile.makedirs(log_dir)
        continue_run = False

    # Copy experiment config file
    shutil.copy(config.__file__, log_dir)

    run_training(continue_run)


if __name__ == '__main__':

    main()
