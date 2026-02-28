import tensorflow as tf
from tensorflow.keras import layers, models, Input

class AttentionLayer(layers.Layer):
    """
    Custom Attention Layer for sequence data focusing on significant temporal features.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

def build_hybrid_model(input_shape=(1, 5, 40), num_classes=8): # Example shape: (timesteps, features, channels)
    """
    Builds the Hybrid CNN + BiLSTM + Attention Model.
    The input_shape heavily depends on how data is grouped (e.g., spectrogram vs aggregated MFCC).
    Here we assume a sequence of feature frames.
    """
    inputs = Input(shape=input_shape)
    
    # 1. 1D CNN Block for spatial feature extraction
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # 2. BiLSTM Block for temporal dependencies
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    
    # 3. Attention Layer
    attention_layer = AttentionLayer()
    x = attention_layer(x)
    
    # 4. Dense Output
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Softmax Classification
    outputs = layers.Dense(num_classes, activation='softmax', name='emotion_output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Hybrid_SER_Model")
    return model

if __name__ == "__main__":
    # Test model shape with a dummy input (timesteps, feature_dim)
    # Assumes flattened frame sequences or raw features
    model = build_hybrid_model(input_shape=(100, 40), num_classes=8)
    model.summary()
