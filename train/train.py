import configparser

from dataGenerator.datagen_v3 import DataGenerator
from hg_models.discriminator import discrim
from hg_models.hg import hgmodel
from train.train_class import train_class


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

        for option in config.options(section):
            params[option] = eval(config.get(section, option))
    return params
params = process_config('./config/config.cfg')
network_params = process_network("./config/hourglass.cfg")
hg = hgmodel(nStack=network_params['nstack'],nModules=network_params['nmodules'])
dis = discrim()
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
trainer = train_class(hg,dis, nstack=network_params['nstack'], batch_size=params['batch_size'],
                              learn_rate=params['lear_rate'], decay=params['decay'],
                              decay_step=params['decay_step'],logdir_train=params['train_log_dir'],
                              logdir_valid=params['valid_log_dir'],name='gan',
                               train_record=train_data,valid_record=valid_data,
                              save_model_dir=params['model_save_path'],
                              resume=params['resume'],
                              gpu=params['gpus'],partnum=network_params['partnum'],
                              val_label=params['valid_label'],train_label=params['label_dir'],
                      human_decay=params['human_decay'],
                     )
trainer.generateModel()
trainer.training_init(nEpochs=params['nepochs'],valStep=params['val_step'] ,showStep=params['show_step'])