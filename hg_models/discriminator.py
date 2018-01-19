
from hg_models.layers.Residual import *

class discrim():
    def __init__(self, nFeat=256,  nModules=2, outputDim=14):


        self.nFeats = nFeat
        self.nModules = nModules
        self.partnum = outputDim

    def hourglass(self,data, n, f, nModual, reuse=False, name=""):

        with tf.variable_scope(name, reuse=reuse):
            pool = tl.layers.MaxPool2d(data, (2, 2), strides=(2, 2), name='pool1')

            up = []
            low = []

            for i in range(nModual):
                if i == 0:
                    tmpup = Residual(data, f, f, name='%s_tmpup_' % (name) + str(i), reuse=reuse)
                    tmplow = Residual(pool, f, f, name='%s_tmplow_' % (name) + str(i), reuse=reuse)
                else:

                    tmpup = Residual(up[i - 1], f, f, name='%s_tmpup_' % (name) + str(i), reuse=reuse)
                    tmplow = Residual(low[i - 1], f, f, name='%s_tmplow_' % (name) + str(i), reuse=reuse)

                up.append(tmpup)
                low.append(tmplow)
            low2_ = []
            if n > 1:
                low2 = self.hourglass(low[-1], n - 1, f, nModual=nModual,
                                 name=name +  "_low2"+"_" + str(n - 1), reuse=reuse)
                low2_.append(low2)
            else:
                for j in range(nModual):

                    if j == 0:
                        tmplow2 = Residual(low[-1], f, f, name='%s_tmplow2_' % (name) + str(j), reuse=reuse)
                    else:
                        tmplow2 = Residual(low2_[j - 1], f, f, name='%s_tmplow2_' % (name) + str(j), reuse=reuse)
                    low2_.append(tmplow2)
            low3_ = []
            for k in range(nModual):
                if k == 0:
                    tmplow3 = Residual(low2_[-1], f, f, name='%s_tmplow3_' % (name) + str(k), reuse=reuse)
                else:
                    tmplow3 = Residual(low3_[k - 1], f, f, name='%s_tmplow3_' % (name) + str(k), reuse=reuse)
                low3_.append(tmplow3)

            up2 = tl.layers.UpSampling2dLayer(low3_[-1], size=[2, 2], is_scale=True, method=1,
                                              name="%s_Upsample" % (name))

            x = tl.layers.ElementwiseLayer(layer=[up[nModual - 1], up2],
                                           combine_fn=tf.add, name="%s_add_layer" % (name))

            return x

    def lin(self,data, numOut, reuse=False, name=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            conv1 = conv_2d(data, numOut, filter_size=(1, 1), strides=(1, 1),
                            name='conv1')
            bn1 = tl.layers.BatchNormLayer(conv1, act=tf.nn.relu, name="bn1", )

            return bn1

    def createModel(self,inputs, reuse=False):
        tl.layers.set_name_reuse(reuse)
        data = tl.layers.InputLayer(inputs, name='discrim_model_input')
        with tf.variable_scope("discrim_model", reuse=reuse):

            conv1 = conv_2d(data, 64, filter_size=(3, 3), strides=(1, 1), padding='SAME', name="conv1")
            bn1 = tl.layers.BatchNormLayer(conv1, name="bn1", act=tf.nn.relu, )
            r1 = Residual(bn1, 64, 128, name="discrim_model_Residual1", reuse=reuse)


            r2 = Residual(r1, 128, 128, name="discrim_model_Residual2", reuse=reuse)

            r3 = Residual(r2, 128, self.nFeats, name="discrim_model_Residual3", reuse=reuse)

        hg = [None]

        ll = [None]
        fc_out = [None]
        with tf.variable_scope('discrim_stage_0', reuse=reuse):
            hg[0] = self.hourglass(r3, n=4, f=self.nFeats, nModual=self.nModules,name="dis_hg", reuse=reuse)
            residual = []
            for i in range(self.nModules):
                if i == 0:
                    tmpres = Residual(hg[0], self.nFeats, self.nFeats, name='tmpres_%d' % (i) , reuse=reuse)
                else:
                    tmpres = Residual(residual[i - 1], self.nFeats, self.nFeats, name='tmpres_%d' % (i), reuse=reuse)
                residual.append(tmpres)

            ll[0] = self.lin( residual[-1], self.nFeats, name="dis_stage_0_lin1" , reuse=reuse)
            fc_out[0] = conv_2d(ll[0], self.partnum, filter_size=(1, 1), strides=(1, 1),
                             name="dis_crim_stage_0_out" )

        return fc_out[-1].outputs