import os
import sys
import time
from tools.img_tf import *
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tools.lr import get_lr
from tools.keypoint_eval import getScore
from tools.ht2coord import getjointcoord
from tools.keypoint_eval import load_annotations
class train_class():
    def __init__(self, model,nstack=4, batch_size=32,learn_rate=2.5e-4, decay=0.96, decay_step=2000,
                 logdir_train="./log/train.log", logdir_valid="./log/test.log",
                name='gan', train_record=None,valid_record=None,save_model_dir="",resume="",gpu=[0],
                 val_label="",train_label="",partnum=14,human_decay=0.96,val_batch_num=10000,beginepoch=0,weight_decay=1e-5
                 ):
        self.batch_size = batch_size
        self.nstack = nstack
        self.learn_r = learn_rate
        self.model = model
        self.lr_decay = decay
        self.lr_decay_step = decay_step
        self.logdir_train = logdir_train
        self.logdir_valid = logdir_valid
        self.name = name
        self.resume = resume
        self.train_record = train_record
        self.valid_record = valid_record
        self.save_dir = save_model_dir
        self.gpu = gpu
        self.cpu = '/cpu:0'
        self.partnum=partnum
        self.joints = ["rShoulder", "rElbow", "rWrist", "lShoulder", "lElbow", "lWrist", "rhip","rknee","rankle",
                       "lhip","lknee","lankle","head","neck"]
        self.val_label = val_label
        self.train_label= train_label
        self.mae = tf.Variable(0, trainable=False, dtype=tf.float32,)
        self.train_mae = tf.Variable(0, trainable=False, dtype=tf.float32,)
        self.human_decay = human_decay
        self.beginepoch = beginepoch
        self.val_batch_num=val_batch_num

        self.weight_decay=weight_decay
    def average_gradients(self,tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        print(zip(*tower_grads))
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # if g:
                    # Add 0 dimension to the gradients to represent the tower.
                    if g is None:
                        continue
                    expanded_g = tf.expand_dims(g, 0)

                    # Append on a 'tower' dimension which we will average over below.
                    grads.append(expanded_g)
            # if flag:
            # Average over the 'tower' dimension.
            if len(grads) == 0:
                continue
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def generateModel(self):
        train_data = self.train_record
        self.train_num = train_data.getN()

        self.h_decay = tf.Variable(1.,  trainable=False,dtype=tf.float32,)

        self.last_learning_rate = tf.Variable(self.learn_r,trainable=False )
        self.global_step = 0
        shuff = tf.random_shuffle(tf.constant(np.arange(0, self.nstack)))
        n = tf.cast(shuff[0], tf.int32)

        with tf.name_scope('lr'):
            # self.lr = tf.train.exponential_decay(self.learn_r, self.train_step, self.lr_decay_step,
            #                                    self.lr_decay,name='learning_rate')

            self.lr = get_lr(self.last_learning_rate, self.global_step, self.lr_decay_step,
                             self.lr_decay, self.h_decay, name='learning_rate')


        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        #self.train_img, self.train_heatmap = train_data.TensorflowBatch()

        with tf.name_scope('inputs'):
            # Shape Input Image - batchSize: None, height: 256, width: 256, channel: 3 (RGB)
            self.train_img = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 3), name='input_img')
            # Shape Ground Truth Map: batchSize x nStack x 64 x 64 x outDim
            self.train_heatmap = tf.placeholder(dtype=tf.float32, shape=(None,  64, 64, self.partnum))

        self.train_output = self.model.build(self.train_img)
        self.loss = 0
        self.last_out = self.train_output[-1]
        for nsta in range(len(self.train_output)):
            self.loss += tf.losses.mean_squared_error(labels=self.train_heatmap, predictions=self.train_output[nsta])

        self.apply_gradient_op = self.optimizer.minimize(self.loss)
        with tf.name_scope('train_heatmap'):
            train_im = self.train_img[n, :, :, :]
            train_im = tf.expand_dims(train_im, 0)

            tf.summary.image(name=('origin_train_img'), tensor=train_im, collections=['train'])

            tout = []
            tgt = []

            for ooo in range(self.partnum):
                hm = self.train_output[-1][n, :, :, ooo]
                hm = tf.expand_dims(hm, -1)
                hm = tf.expand_dims(hm, 0)
                hm = hm * 255
                gt = self.train_heatmap[n, :, :, ooo]

                gt = tf.expand_dims(gt, -1)
                gt = tf.expand_dims(gt, 0)
                gt = gt * 255
                tf.summary.image('ground_truth_%s' % (self.joints[ooo]), tensor=gt,
                                 collections=['train'])
                tf.summary.image('heatmp_%s' % (self.joints[ooo]), hm, collections=['train'])

                tmp = self.train_output[-1][n, :, :, ooo]
                tout.append(tf.cast(tf.equal(tf.reduce_max(tmp), tmp), tf.float32))
                tmp2 = self.train_heatmap[n, :, :, ooo]
                tgt.append(tf.cast(tf.equal(tf.reduce_max(tmp2), tmp2), tf.float32))

            train_gt = tf.add_n(tgt)

            train_gt = tf.expand_dims(train_gt, 0)
            train_gt = tf.expand_dims(train_gt, -1)
            train_hm = tf.add_n(tout)

            train_hm = tf.expand_dims(train_hm, 0)
            train_hm = tf.expand_dims(train_hm, -1)
            tf.summary.image('train_ground_truth', tensor=train_gt, collections=['train'])
            tf.summary.image('train_heatmp', train_hm, collections=['train'])

        with tf.device(self.cpu):

            with tf.name_scope('training'):
                tf.summary.scalar('learning_rate', self.lr, collections=['train'])
                tf.summary.scalar("MAE", self.train_mae, collections=['train'])

            with tf.name_scope('MAE'):
                tf.summary.scalar("MAE", self.mae, collections=['test'])
        self.train_merged = tf.summary.merge_all('train')
        self.valid_merge = tf.summary.merge_all('test')

    def _init_weight(self):
        """ Initialize weights
        """
        print('Session initialization')
        self.Session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.Session.run(tf.global_variables_initializer())

        self.Session.run(tf.local_variables_initializer())
        tl.layers.initialize_global_variables(self.Session)
        print("init done")

    def training_init(self, nEpochs=10, valStep=3000,showStep=10):
        with tf.name_scope('Session'):
                self._init_weight()
                self.saver = tf.train.Saver()
                init = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())

                self.coord = tf.train.Coordinator()
                self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.Session)
                self.Session.run(init)

                if self.resume:
                    print("resume from"+self.resume)
                    self.saver.restore(self.Session, self.resume)
                self.train(nEpochs, valStep,showStep)

    def train(self, nEpochs=10, valStep = 3000,showStep=10 ):
        self.generator = self.train_record.get_batch_generator()
        self.train_num = self.train_record.getN()
        n_step_epoch = int(self.train_num / (self.batch_size * len(self.gpu)))
        self.train_writer = tf.summary.FileWriter(self.logdir_train, self.Session.graph)


        last_lr = self.learn_r
        hm_decay = 1

        for epoch in range( nEpochs):
            self.global_step += 1
            epochstartTime = time.time()
            print('Epoch :' + str(epoch) + '/' + str(nEpochs) + '\n')
            loss = 0
            avg_cost = 0.

            for n_batch in range(n_step_epoch):#n_step_epoch

                percent = ((n_batch + 1) / n_step_epoch) * 100
                num = np.int(20 * percent / 100)
                tToEpoch = int((time.time() - epochstartTime) * (100 - percent) / (percent))
                sys.stdout.write(
                    '\r Train: {0}>'.format("=" * num) + "{0}>".format(" " * (20 - num)) + '||' + str(percent)[
                                                                                                  :4] + '%' + ' -cost: ' + str(
                        loss)[:6] + ' -avg_loss: ' + str(avg_cost)[:5] + ' -timeToEnd: ' + str(tToEpoch) + ' sec.')
                sys.stdout.flush()
                img_train, gt_train, name_train, center_train, scale_train = next(self.generator)
                if n_batch % showStep == 0:
                    # _,__,___,summary,last_lr,train_coord, train_name= self.Session.run\
                        # ([self.apply_hg_grads_,self.apply_discrim_grads_,self.update_K ,self.train_merged,self.lr,self.train_coord,self.train_name_lst[0]],
                        #  feed_dict={self.last_learning_rate : last_lr, self.h_decay:hm_decay})

                    _, train_out, summary, last_lr, = self.Session.run \
                        ([self.apply_gradient_op, self.last_out,self.train_merged, self.lr],
                         # self.train_name_set[0]],
                         feed_dict={self.train_img: img_train, self.train_heatmap: gt_train,
                                    self.last_learning_rate: last_lr, self.h_decay: hm_decay})

                    train_predictions = dict()
                    train_predictions['image_ids'] = []
                    train_predictions['annos'] = dict()
                    coord = self.train_record.recoverFromHm(hm=train_out, center=center_train, scale=scale_train)
                    train_predictions = getjointcoord(coord, name_train, train_predictions)

                    train_return_dict = dict()
                    train_return_dict['error'] = None
                    train_return_dict['warning'] = []
                    train_return_dict['score'] = None
                    train_anno = load_annotations(self.train_label, train_return_dict)
                    train_score = getScore(train_predictions, train_anno, train_return_dict)

                    tmp = self.train_mae.assign(train_score)
                    _ = self.Session.run(tmp)

                    self.train_writer.add_summary(summary, epoch * n_step_epoch + n_batch)
                    self.train_writer.flush()

                else:
                    # _, __, ___,last_lr = self.Session.run([self.apply_hg_grads_,self.apply_discrim_grads_,self.update_K , self.lr],
                    #                          feed_dict={self.last_learning_rate : last_lr, self.h_decay:hm_decay})

                    _, last_lr = self.Session.run(
                        [self.apply_gradient_op, self.lr],
                        feed_dict={self.train_img: img_train, self.train_heatmap: gt_train,
                                   self.last_learning_rate: last_lr, self.h_decay: hm_decay})
                #

            epochfinishTime = time.time()
            # if epoch % 5 == 0:
            #     hm_decay = self.human_decay
            # else:
            #     hm_decay = 1.
            print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(
                int(epochfinishTime - epochstartTime)) + ' sec.' + ' -avg_time/batch: ' + str(
                ((epochfinishTime - epochstartTime) / n_step_epoch))[:4] + ' sec.')
            if epoch % 10 == 0:
                best_model_dir = os.path.join(self.save_dir, self.name + '_' + str(epoch))
                print("epoch "+str(epoch)+", save at " + best_model_dir)
                with tf.name_scope('save'):
                    self.saver.save(self.Session, best_model_dir)

        self.coord.request_stop()
        self.coord.join(self.threads)
        self.Session.close()
        print('Training Done')
