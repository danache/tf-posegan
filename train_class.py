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
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []

            for g, _ in grad_and_vars:
                # if g:
                    # Add 0 dimension to the gradients to represent the tower.

                    expanded_g = tf.expand_dims(g, 0)

                    # Append on a 'tower' dimension which we will average over below.
                    grads.append(expanded_g)
            # if flag:
            # Average over the 'tower' dimension.
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
        generate_time = time.time()
        train_data = self.train_record
        self.train_num = train_data.getN()
        self.train_img_lst = []
        self.train_img_mini_lst = []
        self.train_heatmap_lst = []
        self.train_center_lst = []
        self.train_scale_lst = []
        self.train_name_lst = []




        self.h_decay = tf.Variable(1.,  trainable=False,dtype=tf.float32,)

        self.last_learning_rate = tf.Variable(self.learn_r,trainable=False )

        generate_train_done_time = time.time()
        print('train data generate in ' + str(int(generate_train_done_time - generate_time)) + ' sec.')
        print("train num is %d" % (self.train_num))
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
        #### loss lst
        self.discrim_loss_list = []
        self.discrim_gt_loss_list= []
        self.discrim_pred_loss_list = []
        self.hg_loss_list = []
        self.hg_loss_single_list = []
        #### grad lst
        self.discrim_grads_list = []
        self.hg_grads_list = []
        #### Variable lst
        self.discrim_gt_output_lst = []
        self.hg_output_lst = []
        self.train_for_discrim_pre_lst = []

        self.train_mini_discrim_lst = []
        with tf.variable_scope(tf.get_variable_scope()) as vscope:
            for i in self.gpu:
                with tf.device(("/gpu:%d" % i)):
                    with tf.name_scope('gpu_%d' % ( i)) as scope:
                        train_img, train_mini, train_heatmap, train_center, train_scale, train_name = train_data.getData()
                        self.train_img_lst.append(train_img)
                        self.train_img_mini_lst.append(train_mini)
                        self.train_center_lst.append(train_center)
                        self.train_scale_lst.append(train_scale)
                        self.train_name_lst.append(train_name)
                        self.train_heatmap_lst.append(train_heatmap)


                        # self.train_mini_discrim_lst.append(tf.concat([self.train_heatmap_lst[i][:, -1, :], self.train_img_mini_lst[i]], axis=-1, ))
                        # self.discrim_gt_output_lst.append(self.discriminator.createModel(inputs=self.train_mini_discrim_lst[i],
                        #                                                                  reuse=dis_flag).outputs)
                        #
                        # dis_flag = True
                        # with tf.name_scope('discrim_gt_loss'):
                        #     with tf.device(self.cpu):
                        #         gt_loss = tf.losses.mean_squared_error(labels=self.train_heatmap_lst[i][:,-1,:] ,predictions=self.discrim_gt_output_lst[i])
                        #         self.discrim_gt_loss_list.append(gt_loss)
                        #         with tf.name_scope('training'):
                        #             # print(type(self.loss))
                        #             tf.summary.scalar('discrim_gt_loss_%d' % (i), self.discrim_gt_loss_list[i], collections=['train'])

                        self.hg_output_lst.append(self.hgmodel.createModel(self.train_img_lst[i],reuse=hg_flag).outputs)

                        with tf.name_scope('train_heatmap'):
                            train_im = self.train_img_lst[i][n, :, :, :]
                            train_im = tf.expand_dims(train_im, 0)

                            tf.summary.image(name=('origin_train_img'), tensor=train_im, collections=['train'])

                            tout = []
                            tgt = []

                            for ooo in range(self.partnum):
                                hm = self.hg_output_lst[i][n, self.nstack - 1, :, :, ooo]
                                hm = tf.expand_dims(hm, -1)
                                hm = tf.expand_dims(hm, 0)
                                hm = hm * 255
                                gt = self.train_heatmap_lst[i][n, self.nstack - 1, :, :, ooo]

                                gt = tf.expand_dims(gt, -1)
                                gt = tf.expand_dims(gt, 0)
                                gt = gt * 255
                                tf.summary.image('ground_truth_%s_%d' % (self.joints[ooo], i), tensor=gt,
                                                 collections=['train'])
                                tf.summary.image('heatmp_%s_%d' % (self.joints[ooo], i), hm, collections=['train'])

                                tmp = self.hg_output_lst[i][n, self.nstack - 1, :, :, ooo]
                                tout.append(tf.cast(tf.equal( tf.reduce_max(tmp), tmp),tf.float32))
                                tmp2 = self.train_heatmap_lst[i][n, self.nstack - 1, :, :, ooo]
                                tgt.append(tf.cast(tf.equal( tf.reduce_max(tmp2), tmp2),tf.float32))

                            train_gt = tf.add_n(tgt)

                            train_gt = tf.expand_dims(train_gt, 0)
                            train_gt = tf.expand_dims(train_gt, -1)
                            train_hm= tf.add_n(tout)

                            train_hm = tf.expand_dims(train_hm, 0)
                            train_hm = tf.expand_dims(train_hm, -1)
                            tf.summary.image('train_ground_truth', tensor=train_gt, collections=['train'])
                            tf.summary.image('train_heatmp', train_hm, collections=['train'])

                        hg_flag = True

                        with tf.name_scope('hg_loss'):
                            with tf.device(self.cpu):
                                self.hg_loss_single_list.append(tf.losses.mean_squared_error(labels=self.train_heatmap_lst[i],
                                                                                             predictions=self.hg_output_lst[i]))
                                with tf.name_scope('training'):
                                    # print(type(self.loss))
                                    tf.summary.scalar('hg_loss_%d' % (i), self.hg_loss_single_list[i], collections=['train'])

                        #
                        # tf.get_variable_scope().reuse_variables()
                        #
                        # self.train_for_discrim_pre_lst.append(tf.concat([self.hg_output_lst[i][:, -1, :],self.train_img_mini_lst[i]], axis=-1,) )
                        #
                        # discrim_pred_output = self.discriminator.createModel(inputs=self.train_for_discrim_pre_lst[i], reuse=True)\
                        #     .outputs
                        #
                        # with tf.name_scope('discrim_pred_loss'):
                        #     with tf.device(self.cpu):
                        #         pred_loss = tf.losses.mean_squared_error(labels= self.train_heatmap_lst[i][:,-1,:] ,predictions=discrim_pred_output)
                        #         with tf.name_scope('training'):
                        #             # print(type(self.loss))
                        #             tf.summary.scalar('discrim_pred_loss_%d' % (i), pred_loss, collections=['train'])
                        #
                        #
                        # with tf.name_scope("discrim_all_loss"):
                        #     with tf.device(self.cpu):
                        #         all_loss = tf.subtract(pred_loss ,tf.multiply(self.K , gt_loss))
                        #         self.discrim_loss_list.append(all_loss)
                        #         with tf.name_scope('training'):
                        #             # print(type(self.loss))
                        #             tf.summary.scalar('crmi_all_loss_%d' % (i), all_loss, collections=['train'])
                        #
                        # with tf.name_scope("hg_adv_loss"):
                        #     with tf.device(self.cpu):
                        #         loss_adv = tf.losses.mean_squared_error(self.hg_output_lst[i][:,-1,:],discrim_pred_output)
                        #         hg_all_loss =self.hg_loss_single_list[i] + loss_adv
                        #         self.hg_loss_list.append(hg_all_loss)
                        #         with tf.name_scope('training'):
                        #             # print(type(self.loss))
                        #             tf.summary.scalar('hg_all_loss_%d' % (i),  hg_all_loss, collections=['train'])
                        #
                        # hg_grads = self.optimizer.compute_gradients(loss=self.hg_loss_list[i])
                        # discrim_grads = self.optimizer.compute_gradients(loss=self.discrim_loss_list[i])
                        #
                        # self.discrim_grads_list.append(discrim_grads)
                        # self.hg_grads_list.append(hg_grads)



                        hg_grads = self.optimizer.compute_gradients(loss=self.hg_loss_single_list[i])
                        self.hg_grads_list.append(hg_grads)




        hg_grads_ = self.average_gradients(self.hg_grads_list)
        #discrim_grads_ = self.average_gradients(self.discrim_grads_list)
        ###梯度下降
        self.apply_hg_grads_ = self.optimizer.apply_gradients(hg_grads_)
        # #self.apply_discrim_grads_ = self.optimizer_D.apply_gradients(discrim_grads_)
        # ###更新K值
        # errD_real = tf.reduce_mean(tf.stack(self.discrim_gt_loss_list,axis=0))
        # err_G2 =  tf.reduce_mean(tf.stack(self.discrim_pred_loss_list,axis=0))
        # balance = tf.subtract(self.gamma * errD_real , err_G2 / self.lambda_G)
        #
        # self.update_K = self.K.assign(tf.maximum(tf.minimum(1.,  tf.add(self.K , self.lambda_k * balance)), 0.))
        # with tf.name_scope('training'):
        #     # print(type(self.loss))
        #     tf.summary.scalar('K' , self.K, collections=['train'])
        if self.valid_record:
            valid_data = self.valid_record

            valid_img, valid_mini, valid_ht,self.valid_center, self.valid_scale,self.valid_name = valid_data.getData()
            self.valid_num = valid_data.getN()
            self.validIter = int(self.valid_num / self.batch_size)
            self.valid_output = self.hgmodel.createModel(valid_img,reuse=True).outputs
            generate_valid_done_time = time.time()
            print('train data generate in ' + str(int(generate_valid_done_time - generate_train_done_time )) + ' sec.')
            print("valid num is %d" % (self.valid_num))

            with tf.name_scope('val_heatmap'):

                val_im = valid_img[n, :, :, :]
                val_im = tf.expand_dims(val_im, 0)

                tf.summary.image(name=('origin_valid_img'), tensor=val_im, collections=['test'])

                vout = []
                vgt = []
                for i in range(self.partnum):
                    hm = self.valid_output[n, self.nstack - 1, :,:,i]
                    hm = tf.expand_dims(hm, -1)
                    hm = tf.expand_dims(hm, 0)
                    hm = hm * 255
                    tf.summary.image('heatmp_%s_%d' % (self.joints[i], i), hm, collections=['test'])

                    vout.append(self.valid_output[n, self.nstack - 1, :,:,i])
                    vgt.append(valid_ht[n, self.nstack - 1,:,:, i])
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

        self.train_coord = reverseFromHt(self.hg_output_lst[0], nstack=self.nstack, batch_size=self.batch_size,
                                         num_joint=self.partnum,
                                         scale=self.train_scale_lst[0], center=self.train_center_lst[0], res=[64, 64])

        self.valid_coord = reverseFromHt(self.valid_output, nstack=self.nstack, batch_size=self.batch_size,
                                         num_joint=self.partnum,
                                         scale=self.valid_scale, center=self.valid_center, res=[64, 64])

        self.train_merged = tf.summary.merge_all('train')
        self.valid_merge = tf.summary.merge_all('test')

    def _init_weight(self):
        """ Initialize weights
        """
        print('Session initialization')
        self.Session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
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
            self.valid_writer = tf.summary.FileWriter(self.logdir_valid)

        last_lr = self.learn_r
        hm_decay = 1
        best_val = -1

        valStep = min(valStep, n_step_epoch - 100)
        if self.valid_record:
            valid_iter = int(self.val_batch_num / self.batch_size)
        for epoch in range(self.beginepoch, nEpochs):
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


                    _, summary, last_lr, train_coord, train_name = self.Session.run \
                        ([self.apply_hg_grads_, self.train_merged,self.lr,self.train_coord,self.train_name_lst[0]],
                     feed_dict={self.last_learning_rate : last_lr, self.h_decay:hm_decay})

                    train_predictions = dict()
                    train_predictions['image_ids'] = []
                    train_predictions['annos'] = dict()
                    train_predictions = getjointcoord(train_coord, train_name, train_predictions)

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
                        [self.apply_hg_grads_, self.lr],
                        feed_dict={self.last_learning_rate: last_lr, self.h_decay: hm_decay})
                #
                hm_decay = 1.
                if (n_batch+1) % valStep == 0:

                    if self.valid_record:
                        return_dict = dict()
                        return_dict['error'] = None
                        return_dict['warning'] = []
                        return_dict['score'] = None
                        anno = load_annotations(self.val_label, return_dict)

                        val_begin = time.time()

                        valid_predictions = dict()
                        valid_predictions['image_ids'] = []
                        valid_predictions['annos'] = dict()

                        val_begin_time = time.time()

                        for v in range(valid_iter):  # valid_iter
                            val_percent = ((v + 1) / self.val_batch_num) * 100
                            val_num = np.int(20 * val_percent / 100)
                            val_tToEpoch = int((time.time() - val_begin) * (100 - val_percent) / (val_percent))

                            val_cord, val_name = self.Session.run(
                                [self.valid_coord, self.valid_name, ]
                            )

                            # print(np.array(accuracy_pred).shape)
                            valid_predictions = getjointcoord(val_cord, val_name, valid_predictions)
                            sys.stdout.write(
                                '\r valid {0}>'.format("=" * val_num) + "{0}>".format(" " * (20 - val_num)) + '||' + str(percent)[
                                                                                                  :4][:4] +
                                '%' + ' -cost: ' +
                                ' -timeToEnd: ' + str(val_tToEpoch) + ' sec.')
                            sys.stdout.flush()
                        print("val done in" + str(time.time() - val_begin_time))
                        score = getScore(valid_predictions, anno, return_dict)
                        tmp = self.mae.assign(score)
                        _ = self.Session.run(tmp)

                        print("epoch %d, batch %d ,val score = %f" % (epoch, n_batch, score))
                        tmp = self.mae.assign(score)
                        _ = self.Session.run(tmp)
                        if score > best_val:
                            best_val = score
                            best_model_dir = os.path.join(self.save_dir, self.name + '_' + str(epoch) +
                                                          "_" + str(n_batch) + "_" + (str(score)[:8]))
                            print("get lower loss, save at " + best_model_dir)
                            with tf.name_scope('save'):
                                self.saver.save(self.Session, best_model_dir)
                            hm_decay = 1.
                        '''
                        else:
                            #print("now val loss is not best, restore model from" + best_model_dir)
                            #self.saver.restore(self.Session, best_model_dir)
                            hm_decay = self.human_decay
                        '''
                        valid_summary = self.Session.run([self.valid_merge])

                        self.valid_writer.add_summary(valid_summary[0], epoch * n_step_epoch + n_batch)
                        self.valid_writer.flush()


            epochfinishTime = time.time()
            # if epoch % 5 == 0:
            #     hm_decay = self.human_decay
            # else:
            #     hm_decay = 1.
            print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(
                int(epochfinishTime - epochstartTime)) + ' sec.' + ' -avg_time/batch: ' + str(
                ((epochfinishTime - epochstartTime) / n_step_epoch))[:4] + ' sec.')
            if epoch % 1 == 0:
                best_model_dir = os.path.join(self.save_dir, self.name + '_' + str(epoch))
                print("epoch "+str(epoch)+", save at " + best_model_dir)
                with tf.name_scope('save'):
                    self.saver.save(self.Session, best_model_dir)

        self.coord.request_stop()
        self.coord.join(self.threads)
        self.Session.close()
        print('Training Done')
