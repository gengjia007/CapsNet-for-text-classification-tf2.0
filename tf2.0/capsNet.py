import tensorflow as tf
from capsLayer import CapsLayer

class CapsNet(tf.keras.Model):
    def __init__(self,privateCaps_dim=8,outputCaps_num=7,outputCaps_dim=16,r=3,**kwargs):
        super(CapsNet, self).__init__(name='capsnet',**kwargs)
        self.capsLayer = CapsLayer(privateCaps_dim,outputCaps_num,outputCaps_dim,r)
        self.primaryCapsLength = privateCaps_dim
        self.lamda = 0.5
        self.conv1 = tf.keras.layers.Conv2D(256,(4,25),activation='relu',name='conv1')
        self.conv2 = tf.keras.layers.Conv2D(256,(4,256),strides=(1,128),activation='relu',name='conv2')
        self.batch_size = 1

    def call(self, inputs):
        conv1 = self.conv1(inputs)
        conv1 = tf.reshape(conv1,shape=(-1,conv1.shape[1],conv1.shape[3],conv1.shape[2]))
        conv2 = self.conv2(conv1)
        u_i = tf.reshape(conv2,shape=(-1,1,conv2.shape[1]*conv2.shape[2]*conv2.shape[3]//self.primaryCapsLength,self.primaryCapsLength,1),name='u_i')
        #print(u_i)
        v_j = self.capsLayer(u_i)
        #print(v_j)
        v_k = tf.squeeze(v_j,axis=[2,4],name='v_k')
        #print(v_k)
        v_k_mod = tf.sqrt(tf.reduce_sum(tf.square(v_k),axis=-1))
        #print(v_k_mod)
        return v_k_mod

    def margin_loss(self,y_true, y_pred):
        loss_l = tf.multiply(y_true,tf.square(tf.maximum(0.0,0.9-y_pred)))
        loss_r = self.lamda * tf.multiply(1-y_true,tf.square(tf.maximum(0.0,y_pred-0.1)))
        loss = (tf.reduce_sum(loss_l,axis=[0,1]) + tf.reduce_sum(loss_r,axis=[0,1]))
        return loss
