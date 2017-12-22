import time
import sys
sys.path.append(sys.path[0])
del sys.path[0]
import configparser
from dataGenerator.datagen import DataGenerator
import tensorflow as tf
from hg_models.hg import hgmodel
from train_class import train_class
from predict_class import test_class
import os
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
        if section == 'Network':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
    return params
# FLAGS参数设置
FLAGS = tf.app.flags.FLAGS
# 数据集类型

# 模式：训练、测试
tf.app.flags.DEFINE_string('mode',
                           'train',
                           'train or eval.')
# resume
tf.app.flags.DEFINE_string('resume',
                           '',
                           'restore model path')

if __name__ == '__main__':
    print('--Parsing Config File')
    params = process_config('./config/config.cfg')
    network_params = process_network("./config/hourglass.cfg")
    #network_params = process_network("./config/hgattention.cfg")

    show_step = params["show_step"]
    # test_data = DataGenerator(imgdir=params['train_img_path'], label_dir=params['label_dir'],
    #                            out_record="/media/bnrc2/_backup/ai/mu/test.tfrecords",
    #                            num_txt="/media/bnrc2/_backup/ai/mu/test_num.txt",
    #                            batch_size=params['batch_size'], name="train", is_aug=False,isvalid=True,scale=
    #                            params['scale'])


    model = hgmodel()


    test = test_class(model=model, nstack=network_params['nstack'],
                         test_json="/home/bnrc2/mu/deepcut-pose/python/ly.csv",
                              resume=params['resume'],#/media/bnrc2/_backup/golf/model/tiny_hourglass_21
                              gpu=[0],partnum=network_params['partnum'],
                             )

    test.generateModel()
    test.test_init(img_path="/home/bnrc2/mu/deepcut-pose/python/ly_resize",
                   save_dir="/home/bnrc2/mu/deepcut-pose/python/lygan")


