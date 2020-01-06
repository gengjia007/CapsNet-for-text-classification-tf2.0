import tensorflow as tf 
from tensorflow.keras.layers import Layer

class CapsLayer(Layer):
    def __init__(self, privateCaps_dim,outputCaps_num,outputCaps_dim,r=3):
        super(CapsLayer, self).__init__()
        self.outputCaps_num = outputCaps_num
        self.outputCaps_dim = outputCaps_dim
        self.privateCaps_dim = privateCaps_dim
        self.r = r
        self.epsilon = 1e-9

    def build(self, input_shape):
        self.w_ij = self.add_weight(shape=(1,self.outputCaps_num,int(input_shape[2]),int(input_shape[3]),self.outputCaps_dim),
                                     initializer=tf.keras.initializers.RandomNormal(),
                                     trainable=True,name='w_ij')
    
    def call(self, u_i):
        u_i = tf.reshape(u_i,shape=(-1,1,u_i.shape[1]*u_i.shape[2]*u_i.shape[3]//self.privateCaps_dim,self.privateCaps_dim,1))
        b_ij = tf.zeros([self.outputCaps_num,u_i.shape[2]//1,1,1],name='b_ij')
        for _ in range(self.r):
            u_hat = tf.matmul(self.w_ij,u_i,transpose_a=True,name='u_hat')
            #print(u_hat.shape)
            c_ij = tf.nn.softmax(b_ij,axis=1,name='c_ij')
            #print(c_ij.shape)
            s_j = tf.reduce_sum(tf.multiply(c_ij,u_hat),axis=2,keepdims=True,name='s_j')
            #print(s_j.shape)
            v_j = self.squash(s_j)
            #print(v_j.shape)
            update = tf.reduce_mean(tf.matmul(u_hat,v_j,transpose_a=True),axis=0)
            #print(update)
            b_ij += update
            #print(b_ij)
        
        return v_j
    
    def squash(self,vector):
        '''Squashing function corresponding to Eq. 1
        Args:
            vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
        Returns:
            A tensor with the same shape as vector but squashed in 'vec_len' dimension.
        '''
        vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + self.epsilon)
        vec_squashed = scalar_factor * vector  # element-wise
        return(vec_squashed)
