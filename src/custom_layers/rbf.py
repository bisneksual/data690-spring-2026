import tensorflow as tf

class RBFLayer(tf.keras.layers.Layer):
    def __init__(self,units,gamma,**kwargs):
        super(RBFLayer,self).__init__(**kwargs)
        self.units=units
        self.gamma = gamma
    
    def build(self,input_shape):
        #Centers (mu) are trainable weights
        self.mu = self.add_weight(
            name='mu',
            shape=(int(input_shape[-1]),self.units),
            initializer='uniform',
            trainable=True
        )
        super(RBFLayer,self).build(input_shape)
    
    def call(self,inputs):
        #Calculate Euclidian distance squared
        diff = tf.expand_dims(inputs,axis=-1) - self.mu
        l2 = tf.reduce_sum(pow(diff,2),axis=1,keepdims=False)
        # Gaussian activation
        return tf.math.exp(-1 * self.gamma * l2)
    
    def compute_output_shape(self,input_shape):
        return (input_shape[0],self.units)