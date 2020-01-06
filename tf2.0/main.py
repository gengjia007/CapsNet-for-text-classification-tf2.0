import tensorflow as tf 
import numpy as np 
import utils
from capsNet import CapsNet

if __name__ == "__main__":
    vec,label,map_label = utils.get_vector('train',12,None)
    vec = np.array(vec).astype(np.float)
    vec_test,label_test,_ = utils.get_vector('test',12,map_label)
    vec_test = np.array(vec_test).astype(np.float)
    vec = np.array(vec.reshape(-1,12,25,1)).astype(np.float32)
    vec_test = vec_test.reshape(-1,12,25,1)

    caps_model = CapsNet()
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=3, min_lr=1e-5)
    #tensorBoard = tf.keras.callbacks.TensorBoard(log_dir='logs')
    caps_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss=caps_model.margin_loss,metrics=['accuracy'])
    caps_model.fit(vec,label,batch_size=32,epochs=50,validation_data=(vec_test,label_test),callbacks=[earlyStopping,reduce_lr])

