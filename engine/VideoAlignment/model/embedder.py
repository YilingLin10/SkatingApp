import tensorflow as tf
import numpy as np

# Embedding Model
class Embedder(tf.keras.Model):
  def __init__(self, embedding_size, normalize_embeddings,
               num_context_steps):
    super().__init__()

    # Will download pre-trained ResNet50V2 here
    base_model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False,
                                        weights='imagenet',
                                        pooling='max')
    layer = 'conv4_block3_out'
    self.num_context_steps = num_context_steps
    self.base_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(layer).output)
    self.conv_layers = [tf.keras.layers.Conv3D(256, 3, padding='same')
                        for _ in range(2)]
    self.bn_layers = [tf.keras.layers.BatchNormalization()
                        for _ in range(2)]

    self.fc_layers = [tf.keras.layers.Dense(256,
                                            activation=tf.nn.relu) for _ in range(2)]
    
    self.embedding_layer = tf.keras.layers.Dense(embedding_size)
    self.normalize_embeddings = normalize_embeddings
    self.dropout = tf.keras.layers.Dropout(0.1)
  
  def call(self, frames, training):
    batch_size, _, h,  w, c = frames.shape
    frames = tf.reshape(frames,[-1, h, w, c])

    x = self.base_model(frames , training=training)
    _, h,  w, c = x.shape
    x = tf.reshape(x, [-1, self.num_context_steps, h, w, c])

    x = self.dropout(x)

    for conv_layer, bn_layer in zip(self.conv_layers,
                                    self.bn_layers):
      x = conv_layer(x)
      x = bn_layer(x)
      x = tf.nn.relu(x)
             
    x = tf.reduce_max(x, [1, 2, 3])

    _, c = x.shape
    x = tf.reshape(x, [batch_size, -1, c]) 
    
    for fc_layer in self.fc_layers:
      x = self.dropout(x)
      x = fc_layer(x)

    x = self.embedding_layer(x)
    
    if self.normalize_embeddings:
      x = tf.nn.l2_normalize(x, axis=-1)
    
    return x