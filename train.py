import configparser

from dataGenerator.datagen_v3 import DataGenerator
from hg_models.ian_hourglass import hourglassnet
from train_class import train_class

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

        for option in config.options(section):
            params[option] = eval(config.get(section, option))
    return params
params = process_config('./config/config.cfg')
network_params = process_network("./config/hourglass.cfg")
###
model = hourglassnet()
###
train_data = DataGenerator(imgdir=params["train_img_path"], txt="/media/bnrc2/_backup/dataset/aiclg/data.txt", batch_size=params['batch_size'], is_aug=False,
                           joints_name=
                           params["joints"])  # , refine_num = 10000)
valid_data = DataGenerator(imgdir=params["valid_img_path"], txt="/media/bnrc2/_backup/dataset/aiclg/valid.txt", batch_size=params['batch_size'], is_aug=False,
                           joints_name=
                           params["joints"],isTraing=False)
valid_gen = valid_data.get_batch_generator()
img_valid, gt_valid,  name_valid, center_valid, scale_valid = next(
                                valid_gen)
print(img_valid.shape)
print(gt_valid.shape)
# print(name_valid.shape)
# print(center_valid.shape)
# print(scale_valid.shape)
# exit(-1)
train_log_dir = "/media/bnrc2/_backup/log/self-gan/GAN2GPU/train.log"
valid_log_dir = "/media/bnrc2/_backup/log/self-gan/GAN2GPU/valid.log"
trainer = train_class(model, nstack=network_params['nstack'], batch_size=params['batch_size'],
                              learn_rate=params['lear_rate'], decay=params['decay'],
                              decay_step=params['decay_step'],logdir_train=train_log_dir,
                              logdir_valid=valid_log_dir,name='gan',
                               train_record=train_data,valid_record=valid_data,
                              save_model_dir=params['model_save_path'],
                              # resume="/media/bnrc2/_backup/models/gan2gpu/gan_1",
                              gpu=[0],partnum=network_params['partnum'],
                              val_label=params['valid_label'],train_label=params['label_dir'],
                      human_decay=params['human_decay'],
                     )
trainer.generateModel()
trainer.training_init(nEpochs=params['nepochs'],valStep=params['val_step'] ,showStep=params['show_step'])