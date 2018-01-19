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
    def __init__(self, hgmodel,discrim, nstack=4, batch_size=32,learn_rate=2.5e-4, decay=0.96, decay_step=2000,gamma=0.5,
                 lambda_G=0.0001,lambda_K=0.001,
                 logdir_train="./log/train.log", logdir_valid="./log/test.log",
                name='gan', train_record=None,valid_record=None,save_model_dir="",resume="",gpu=[0],
                 val_label="",train_label="",partnum=14,human_decay=0.96,val_batch_num=10000,beginepoch=0,weight_decay=1e-5
                 ):
        self.batch_size = batch_size
        self.nstack = nstack
        self.learn_r = learn_rate

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
        self.hgmodel = hgmodel
        self.discriminator = discrim
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
        self.gamma = gamma
        self.lambda_G = lambda_G
        self.lambda_k = lambda_K
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
        self.optimizer_D = tf.train.RMSPropOptimizer(learning_rate=8e-5)
        dis_flag = False
        hg_flag = False
        self.K = tf.Variable(0., trainable=False, dtype=tf.float32, )

        ###### data list ######
        self.train_img_lst = []
        self.train_img_mini_lst = []
        self.train_heatmap_lst = []

        ##### mid output list #####
        self.concat_trainmini_heatmap = []
        self.concat_generator_heatmap = []
        self.discriminator_groundTruth_output = []
        self.discriminator_generator_output = []

        self.generator_output = []
        ##### loss list ######
        self.discriminator_groundTruth_loss = []
        self.discriminator_generator_loss = []
        self.discriminator_all_loss = []

        self.generator_loss = []
        self.generator_all_loss = []
        ##### gradient ####
        self.generator_gradient = []
        self.discriminator_gradient = []
        ###################################


        with tf.variable_scope(tf.get_variable_scope()) as vscope:
            for i in self.gpu:
                with tf.device(("/gpu:%d" % i)):
                    with tf.name_scope('gpu_%d' % ( i)) as scope:
                        ####### get train data #######
                        train_img, train_mini, train_heatmap = train_data.TensorflowBatch()
                        self.train_img_lst.append(train_img)
                        self.train_img_mini_lst.append(train_mini)
                        self.train_heatmap_lst.append(train_heatmap)

                        ####### concatHeatMap with gt and minimap #######

                        self.concat_trainmini_heatmap.append(tf.concat([self.train_heatmap_lst[i],
                                                                        self.train_img_mini_lst[i]], axis=-1, ))
                        # tmp = tf.placeholder(shape=[None,64,64,17])
                        ####### groundTruth output #######
                        self.discriminator_groundTruth_output.append(self.discriminator.createModel(inputs=self.concat_trainmini_heatmap[i],
                                                                                             reuse=dis_flag))
                        # self.discriminator_groundTruth_output.append(self.discriminator.createModel(inputs=
                        #                                                                             tmp,
                        #                                                                      reuse=dis_flag))
                        dis_flag = True
                        ####### calculate discrim_gtt_loss ######
                        with tf.name_scope('discrim_gt_loss'):
                            with tf.device(self.cpu):
                                gt_loss = tf.losses.mean_squared_error(labels=self.train_heatmap_lst[i] ,
                                                                       predictions=self.discriminator_groundTruth_output[i])
                                self.discriminator_groundTruth_loss.append(gt_loss)
                                with tf.name_scope('training'):
                                    # print(type(self.loss))
                                    tf.summary.scalar('discrim_gt_loss_%d' % (i), self.discriminator_groundTruth_loss[i],
                                                      collections=['train'])

                        ###### get Hourglass output###
                        self.generator_output.append(self.hgmodel.createModel(self.train_img_lst[i],reuse=hg_flag))
                        hg_flag = True

                        ####### generate plot heatmap ########

                        with tf.name_scope('train_heatmap'):
                            train_im = self.train_img_lst[i][n, :, :, :]
                            train_im = tf.expand_dims(train_im, 0)

                            tf.summary.image(name=('origin_train_img'), tensor=train_im, collections=['train'])

                            tout = []
                            tgt = []

                            for ooo in range(self.partnum):
                                hm = self.generator_output[i][-1][n, :, :, ooo]
                                hm = tf.expand_dims(hm, -1)
                                hm = tf.expand_dims(hm, 0)
                                hm = hm * 255
                                gt = self.train_heatmap_lst[i][n,  :, :, ooo]

                                gt = tf.expand_dims(gt, -1)
                                gt = tf.expand_dims(gt, 0)
                                gt = gt * 255
                                tf.summary.image('ground_truth_%s_%d' % (self.joints[ooo], i), tensor=gt,
                                                 collections=['train'])
                                tf.summary.image('heatmp_%s_%d' % (self.joints[ooo], i), hm, collections=['train'])

                                tmp = self.generator_output[i][-1][n, :, :, ooo]
                                tout.append(tf.cast(tf.equal( tf.reduce_max(tmp), tmp),tf.float32))
                                tmp2 = self.train_heatmap_lst[i][n,:, :, ooo]
                                tgt.append(tf.cast(tf.equal( tf.reduce_max(tmp2), tmp2),tf.float32))

                            train_gt = tf.add_n(tgt)

                            train_gt = tf.expand_dims(train_gt, 0)
                            train_gt = tf.expand_dims(train_gt, -1)
                            train_hm= tf.add_n(tout)

                            train_hm = tf.expand_dims(train_hm, 0)
                            train_hm = tf.expand_dims(train_hm, -1)
                            tf.summary.image('train_ground_truth', tensor=train_gt, collections=['train'])
                            tf.summary.image('train_heatmp', train_hm, collections=['train'])

                       ######## calculate hourglass loss #####

                        with tf.name_scope('hg_loss'):
                            with tf.device(self.cpu):
                                loss = 0
                                for nsta in range(self.nstack):
                                    loss += tf.losses.mean_squared_error(labels=self.train_heatmap_lst[i],
                                                                                             predictions=
                                                                                             self.generator_output[i][nsta])
                                self.generator_loss.append(loss)


                        with tf.name_scope('training'):
                            # print(type(self.loss))
                            tf.summary.scalar('hg_loss_%d' % (i), self.generator_loss[i], collections=['train'])


                        tf.get_variable_scope().reuse_variables()

                        ### concat generator output and miniimg###

                        self.concat_generator_heatmap.append(tf.concat([self.generator_output[i][-1],
                                                                        self.train_img_mini_lst[i]], axis=-1,) )

                        #### foward propogation ###
                        self.discriminator_generator_output.append(self.discriminator.createModel
                                                                   (inputs=self.concat_generator_heatmap[i],
                                                                             reuse=True))

                        with tf.name_scope('discrim_pred_loss'):
                            with tf.device(self.cpu):
                                self.discriminator_generator_loss.append(tf.losses.mean_squared_error(labels= self.train_heatmap_lst[i] ,
                                                                         predictions=self.discriminator_generator_output[i]))
                                with tf.name_scope('training'):
                                    # print(type(self.loss))
                                    tf.summary.scalar('discrim_pred_loss_%d' % (i),self.discriminator_generator_loss[i],
                                                      collections=['train'])


                        with tf.name_scope("discrim_all_loss"):
                            with tf.device(self.cpu):
                                self.discriminator_all_loss.append(tf.subtract(self.discriminator_generator_loss[i] ,
                                                                               tf.multiply(self.K , gt_loss)))
                                with tf.name_scope('training'):
                                    # print(type(self.loss))
                                    tf.summary.scalar('crmi_all_loss_%d' % (i), self.discriminator_all_loss[i],
                                                      collections=['train'])

                        with tf.name_scope("hg_adv_loss"):
                            with tf.device(self.cpu):
                                loss_adv = tf.losses.mean_squared_error(self.generator_output[i][-1],
                                                                        self.discriminator_generator_output[i])
                                self.generator_all_loss.append(tf.add(self.generator_loss[i],
                                                                      tf.multiply(self.lambda_G,loss_adv)))
                                with tf.name_scope('training'):
                                    # print(type(self.loss))
                                    tf.summary.scalar('hg_all_loss_%d' % (i), self.generator_all_loss[i], collections=['train'])

                        hg_grads = self.optimizer.compute_gradients(loss=self.generator_all_loss[i])
                        discrim_grads = self.optimizer.compute_gradients(loss=self.discriminator_all_loss[i])

                        self.generator_gradient.append(hg_grads)
                        self.discriminator_gradient.append(discrim_grads)


        generator_grads_ = self.average_gradients(self.generator_gradient)
        discrim_grads_ = self.average_gradients(self.discriminator_gradient)
        ###梯度下降
        self.apply_hg_grads_ = self.optimizer.apply_gradients(generator_grads_)
        self.apply_discrim_grads_ = self.optimizer_D.apply_gradients(discrim_grads_)


        ###更新K值

        errD_real = tf.reduce_mean(tf.stack(self.discriminator_groundTruth_loss,axis=0))
        err_G2 =  tf.reduce_mean(tf.stack(self.discriminator_generator_loss,axis=0))
        balance = tf.subtract(self.gamma * errD_real , err_G2 / self.lambda_G)

        self.update_K = self.K.assign(tf.maximum(tf.minimum(1.,  tf.add(self.K , self.lambda_k * balance)), 0.))
        with tf.name_scope('training'):
            # print(type(self.loss))
            tf.summary.scalar('K' , self.K, collections=['train'])


        if self.valid_record:

            self.valid_img = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 3), name='input_img')
            # Shape Ground Truth Map: batchSize x nStack x 64 x 64 x outDim
            self.valid_ht = tf.placeholder(dtype=tf.float32, shape=(None, 64, 64, self.partnum))

            self.valid_output = self.hgmodel.createModel(self.valid_img,reuse=True)[-1]

            with tf.name_scope('val_heatmap'):

                val_im = self.valid_img[n, :, :, :]
                val_im = tf.expand_dims(val_im, 0)

                tf.summary.image(name=('origin_valid_img'), tensor=val_im, collections=['test'])

                vout = []
                vgt = []
                for i in range(self.partnum):
                    hm = self.valid_output[n, :,:,i]
                    hm = tf.expand_dims(hm, -1)
                    hm = tf.expand_dims(hm, 0)
                    hm = hm * 255
                    tf.summary.image('heatmp_%s_%d' % (self.joints[i], i), hm, collections=['test'])

                    vout.append(self.valid_output[n, :,:,i])
                    vgt.append(self.valid_ht[n,:,:, i])
                val_hm = tf.add_n(vout)
                val_gt = tf.add_n(vgt)

                val_hm = tf.expand_dims(val_hm, 0)
                val_hm = tf.expand_dims(val_hm, -1)
                val_gt = tf.expand_dims(val_gt, 0)
                val_gt = tf.expand_dims(val_gt, -1)

                tf.summary.image('valid_ground_truth', tensor=val_gt,collections=['test'])
                tf.summary.image('valid_heatmp' , val_hm, collections=['test'])


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
            with tf.device(self.gpu[0]):
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

        n_step_epoch = int(self.train_num / (self.batch_size * len(self.gpu)))
        self.train_writer = tf.summary.FileWriter(self.logdir_train, self.Session.graph)
        if self.valid_record:
            self.valid_gen = self.valid_record.get_batch_generator()
            self.valid_writer = tf.summary.FileWriter(self.logdir_valid)

        last_lr = self.learn_r
        hm_decay = 1
        best_val = -1
        valid_iter = 1000
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

                if n_batch % showStep == 0:
                    # _,__,___,summary,last_lr,train_coord, train_name= self.Session.run\
                        # ([self.apply_hg_grads_,self.apply_discrim_grads_,self.update_K ,self.train_merged,self.lr,self.train_coord,self.train_name_lst[0]],
                        #  feed_dict={self.last_learning_rate : last_lr, self.h_decay:hm_decay})

                    _, ___,____,summary, last_lr,  = self.Session.run \
                        ([self.apply_hg_grads_, self.apply_discrim_grads_,self.update_K,self.train_merged, self.lr],
                         feed_dict={self.last_learning_rate: last_lr, self.h_decay: hm_decay})


                    self.train_writer.add_summary(summary, epoch * n_step_epoch + n_batch)
                    self.train_writer.flush()

                else:
                    # _, __, ___,last_lr = self.Session.run([self.apply_hg_grads_,self.apply_discrim_grads_,self.update_K , self.lr],
                    #                          feed_dict={self.last_learning_rate : last_lr, self.h_decay:hm_decay})

                    _, __,___,last_lr = self.Session.run(
                        [self.apply_hg_grads_, self.apply_discrim_grads_,self.update_K,self.lr],
                        feed_dict={self.last_learning_rate: last_lr, self.h_decay: hm_decay})
                #
                hm_decay = 1.
                if (n_batch+1) % valStep == 0:

                    if self.valid_record:
                        valid_return_dict = dict()
                        valid_return_dict['error'] = None
                        valid_return_dict['warning'] = []
                        valid_return_dict['score'] = None

                        valid_anno = load_annotations(self.val_label, valid_return_dict)
                        val_begin = time.time()

                        valid_predictions = dict()
                        valid_predictions['image_ids'] = []
                        valid_predictions['annos'] = dict()


                        for v in range(valid_iter):  # valid_iter

                            img_valid, gt_valid,  name_valid, center_valid, scale_valid = next(
                                self.valid_gen)

                            val_percent = ((v + 1) / self.val_batch_num) * 100
                            val_num = np.int(20 * val_percent / 100)
                            val_tToEpoch = int((time.time() - val_begin) * (100 - val_percent) / (val_percent))

                            valid_out = self.Session.run(
                                self.valid_output,feed_dict={self.valid_img:img_valid,self.valid_ht:
                                                                                 gt_valid}
                            )

                            # print(np.array(accuracy_pred).shape)
                            #valid_predictions = getjointcoord(val_cord, val_name, valid_predictions)
                            vadli_coord = self.train_record.recoverFromHm(hm=valid_out, center=center_valid,
                                                                    scale=scale_valid)
                            valid_predictions = getjointcoord(vadli_coord, name_valid, valid_predictions)


                            sys.stdout.write(
                                '\r valid {0}>'.format("=" * val_num) + "{0}>".format(" " * (20 - val_num)) + '||' + str(percent)[
                                                                                                  :4][:4] +
                                '%' + ' -cost: ' +
                                ' -timeToEnd: ' + str(val_tToEpoch) + ' sec.')
                            sys.stdout.flush()
                        score = getScore(valid_predictions, valid_anno, valid_return_dict)
                        tmp = self.mae.assign(score)
                        _,valid_summary = self.Session.run([tmp,self.valid_merge],feed_dict={self.valid_img:img_valid,self.valid_ht:
                                                                                 gt_valid} )

                        self.valid_writer.add_summary(valid_summary, epoch * n_step_epoch + n_batch)
                        self.valid_writer.flush()

                        if score > best_val:
                            best_val = score
                            best_model_dir = os.path.join(self.save_dir, self.name + '_' + str(epoch) +
                                                          "_" + str(n_batch) + "_" + (str(score)[:8]))
                            print("get lower loss, save at " + best_model_dir)
                            with tf.name_scope('save'):
                                self.saver.save(self.Session, best_model_dir)
                            hm_decay = 1.

                        # else:
                        #     #print("now val loss is not best, restore model from" + best_model_dir)
                        #     #self.saver.restore(self.Session, best_model_dir)
                        #     hm_decay = self.human_decay




            epochfinishTime = time.time()
            # if epoch % 5 == 0:
            #     hm_decay = self.human_decay
            # else:
            #     hm_decay = 1.
            print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(
                int(epochfinishTime - epochstartTime)) + ' sec.' + ' -avg_time/batch: ' + str(
                ((epochfinishTime - epochstartTime) / n_step_epoch))[:4] + ' sec.')
            if epoch % 2 == 0:
                best_model_dir = os.path.join(self.save_dir, self.name + '_' + str(epoch))
                print("epoch "+str(epoch)+", save at " + best_model_dir)
                with tf.name_scope('save'):
                    self.saver.save(self.Session, best_model_dir)

        self.coord.request_stop()
        self.coord.join(self.threads)
        self.Session.close()
        print('Training Done')
