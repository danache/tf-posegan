
from hg_models.layers.Residual import *



class hgmodel():
    def __init__(self, nFeat=256, nStack=4, nModules=2, outputDim=14,npool=4):
        self.nStack = nStack
        self.nFeats = nFeat
        self.nModules = nModules
        self.partnum = outputDim
        self.npool = npool

    def hourglass(self,data, n, f, nModual, reuse=False, name=""):

        # with mx.name.Prefix("%s_%s_" % (name, suffix)):
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
        data = tl.layers.InputLayer(inputs, name='input')
        with tf.variable_scope("hg_model", reuse=reuse):

            conv1 = conv_2d(data, 64, filter_size=(7, 7), strides=(2, 2), padding='SAME', name="conv1")
            bn1 = tl.layers.BatchNormLayer(conv1, name="bn1", act=tf.nn.relu, )
            r1 = Residual(bn1, 64, 128, name="Residual1", reuse=reuse)
            pool = tl.layers.MaxPool2d(r1, (2, 2), strides=(2, 2), name="pool1")

            r2 = Residual(pool, 128, 128, name="Residual2", reuse=reuse)

            r3 = Residual(r2, 128, self.nFeats, name="Residual3", reuse=reuse)



        hg = [None] * self.nStack

        ll = [None] * self.nStack
        fc_out = [None] * self.nStack
        c_1 = [None] * self.nStack
        c_2 = [None] * self.nStack
        sum_ = [None] * self.nStack
        resid = dict()
        out = []

        with tf.variable_scope("hg_stack", reuse=reuse):

            hg[0] = self.hourglass(r3, n=4, f=self.nFeats, name="stage_0_hg", nModual=self.nModules,reuse=reuse)

            resid["stage_0"] = []
            for i in range(self.nModules):
                if i == 0:
                    tmpres = Residual(hg[0], self.nFeats, self.nFeats, name='stage_0tmpres_%d' % (i), reuse=reuse)
                else:
                    tmpres = Residual(resid["stage_0"][i - 1], self.nFeats, self.nFeats, name='stage_0tmpres_%d' % (i),
                                      reuse=reuse)
                resid["stage_0"].append(tmpres)


            ll[0] = self.lin(resid["stage_0"][-1], self.nFeats, name="stage_0_lin1", reuse=reuse)
            fc_out[0] = conv_2d(ll[0], self.partnum, filter_size=(1, 1), strides=(1, 1),
                                name="stage_0_out")
            out.append(fc_out[0])
            if self.nStack > 1:
                c_1[0] = conv_2d(ll[0], self.nFeats, filter_size=(1, 1), strides=(1, 1),
                                 name="stage_0_conv1")

                c_2[0] = conv_2d(c_1[0], self.nFeats, filter_size=(1, 1), strides=(1, 1),
                                 name="stage_0_conv2")
                sum_[0] = tl.layers.ElementwiseLayer(layer=[r3, c_1[0], c_2[0]],
                                                     combine_fn=tf.add, name="stage_0_add_n")

            for i in range(1, self.nStack - 1):
                with tf.variable_scope('stage_%d' % (i)):

                    hg[i] = self.hourglass(sum_[i - 1], n=4, f=self.nFeats, nModual=self.nModules,name="stage_%d_hg" % (i), reuse=reuse)

                    resid["stage_%d"%i] = []
                    for j in range(self.nModules):
                        if j == 0:
                            tmpres = Residual(hg[i], self.nFeats, self.nFeats, name='stage_%d_tmpres_%d' % (i,j),
                                              reuse=reuse)
                        else:
                            tmpres = Residual(resid["stage_%d"%i][j - 1], self.nFeats, self.nFeats,
                                              name='stage_%d_tmpres_%d' % (i, j),
                                              reuse=reuse)
                        resid["stage_%d"%i].append(tmpres)

                    ll[i] = self.lin(resid["stage_%d"%i][-1], self.nFeats, name="stage_%d_lin" % (i), reuse=reuse)
                    fc_out[i] = conv_2d(ll[i], self.partnum, filter_size=(1, 1), strides=(1, 1),
                                        name="stage_%d_out" % (i))
                    out.append(fc_out[i])

                    c_1[i] = conv_2d(ll[i], self.nFeats, filter_size=(1, 1), strides=(1, 1),
                                     name="stage_%d_conv1" % (i))

                    c_2[i] = conv_2d(c_1[i], self.nFeats, filter_size=(1, 1), strides=(1, 1),
                                     name="stage_%d_conv2" % (i))
                    sum_[i] = tl.layers.ElementwiseLayer(layer=[sum_[i - 1], c_1[i], c_2[i]],
                                                         combine_fn=tf.add, name="stage_%d_add_n" % (i))
            with tf.variable_scope('stage_%d' % (self.nStack - 1)):
                hg[self.nStack - 1] = self.hourglass(sum_[self.nStack - 2], n=4, f=self.nFeats,nModual=self.nModules,
                                                     name="stage_%d_hg" % (self.nStack - 1), reuse=reuse)
                residual = []
                for j in range(self.nModules):
                    if j == 0:
                        tmpres = Residual(hg[self.nStack - 1], self.nFeats, self.nFeats, name='stage_%d_tmpres_%d' % (self.nStack - 1, j),
                                          reuse=reuse)
                    else:
                        tmpres = Residual(residual[j - 1], self.nFeats, self.nFeats,
                                          name='stage_%d_tmpres_%d' % (self.nStack - 1, j),
                                          reuse=reuse)
                    residual.append(tmpres)

                ll[self.nStack - 1] = self.lin(residual[-1], self.nFeats,
                                               name="stage_%d_lin1" % (self.nStack - 1), reuse=reuse)
                fc_out[self.nStack - 1] = conv_2d(ll[self.nStack - 1], self.partnum, filter_size=(1, 1),
                                                  strides=(1, 1),
                                                  name="stage_%d_out" % (self.nStack - 1))
                out.append(fc_out[self.nStack - 1])
        # end = out[0]
        end = tl.layers.StackLayer(out, axis=1, name='final_output')

        return end

