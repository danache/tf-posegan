import configparser
import os

from dataGenerator.datagen_HG import DataGenerator
from hg_models.ian_hourglass import hourglassnet
from train_class_HG_Tensor import train_class

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
model = hourglassnet()
train_data = DataGenerator(imgdir=params["train_img_path"], txt="/media/bnrc2/_backup/dataset/aiclg/data.txt", batch_size=params['batch_size'], is_aug=False,
                           joints_name=
                           params["joints"])  # , refine_num = 10000)

train_log_dir = "/media/bnrc2/_backup/log/test_gan_gpu/SGPU/train.log"
valid_log_dir = "/media/bnrc2/_backup/log/test_gan_gpu/SGPU/valid.log"

trainer = train_class(model, nstack=network_params['nstack'], batch_size=params['batch_size'],
                              learn_rate=params['lear_rate'], decay=params['decay'],
                              decay_step=params['decay_step'],logdir_train=train_log_dir,
                              logdir_valid=valid_log_dir,name='gan',
                               train_record=train_data,valid_record=None,
                              save_model_dir=params['model_save_path'],
                              gpu=[0],partnum=network_params['partnum'],
                              val_label=params['valid_label'],train_label=params['label_dir'],
                      human_decay=params['human_decay'],
                     )
trainer.generateModel()
trainer.training_init(nEpochs=params['nepochs'],valStep=params['val_step'] ,showStep=params['show_step'])