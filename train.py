import time

import configparser

import tensorflow as tf

from train_class import train_class
from hg_models.discriminator import discrim
from hg_models.hg import hgmodel
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def process_config(conf_file):
    params = {}
    config = configparser.ConfigParser()
    config.read(conf_file)
    for section in config.sections():
        if section == 'DataSetHG':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'log':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Saver':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Training setting':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
    return params

def process_network(conf_file):
    params = {}
    config = configparser.ConfigParser()
    config.read(conf_file)
    for section in config.sections():
        if section == 'Network':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
    return
params = process_config('./config/config.cfg')
network_params = process_network("./config/hourglass.cfg")


trainer = train_class(hgmodel,discrim, nstack=network_params['nstack'], batch_size=params['batch_size'],
                              learn_rate=params['lear_rate'], decay=params['decay'],
                              decay_step=params['decay_step'],logdir_train=params['train_log_dir'],
                              logdir_valid=params['valid_log_dir'],name='hourglass_8',
                              # train_record=train_data,valid_record=valid_data,
                              save_model_dir=params['model_save_path'],
                              resume=params['resume'],#/media/bnrc2/_backup/golf/model/tiny_hourglass_21
                              gpu=params['gpus'],partnum=network_params['partnum'],
                              val_label=params['valid_label'],human_decay=params['human_decay'],beginepoch=1
                     )
trainer.generateModel()
trainer.training_init(nEpochs=params['nepochs'],valStep=params['val_step'] )