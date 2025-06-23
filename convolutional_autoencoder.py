import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, \
    BatchNormalization, Activation, ZeroPadding1D, Cropping1D, Flatten, Dense, Reshape
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import umap
import pandas as pd


# Load the MATLAB file
mat_data = loadmat("/home/eloy/Code/DeepSup/Analysis/DeepSupDB/Original/filtRipples.mat")  
filt_ripples = mat_data["filtRipples"]

# Convert to float32 (important for TensorFlow) and reshape for Conv1D
filt_ripples = np.array(filt_ripples, dtype=np.float32).reshape(filt_ripples.shape[0], filt_ripples.shape[1], 1)



# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


def build_convolutional_autoencoder(input_shape=(127, 1), encoding_dim=16):
    """
    Build autoencoder with padding to handle the dimension mismatch and
    with a dense bottleneck layer for dimensionality reduction.
    """
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Pad input from 127 to 128
    padded = ZeroPadding1D(padding=(0, 1))(input_layer)
    
    # Encoder
    x = Conv1D(32, 3, padding='same')(padded)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2, padding='same')(x)  # 128 -> 64
    
    x = Conv1D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2, padding='same')(x)  # 64 -> 32
    
    x = Conv1D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2, padding='same')(x)  # 32 -> 16
    
    # Flatten before dense layers
    x = Flatten()(x)
    
    # Dense layer to reduce dimensions
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Bottleneck layer (dense)
    encoded = Dense(encoding_dim, name='bottleneck')(x)
    
    # Start of decoder - dense layers
    x = Dense(128)(encoded)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Dense layer to match flattened dimensions
    x = Dense(16 * 128)(x)  # Match dimensions from last conv layer
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Reshape back to convolutional format
    x = Reshape((16, 128))(x)
    
    # Continue with convolutional decoder
    x = Conv1D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling1D(2)(x)  # 16 -> 32
    
    x = Conv1D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling1D(2)(x)  # 32 -> 64
    
    x = Conv1D(32, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling1D(2)(x)  # 64 -> 128
    
    # Output layer
    x = Conv1D(1, 3, padding='same', activation='linear')(x)
    
    # Crop back to original size (128 -> 127)
    decoded = Cropping1D(cropping=(0, 1))(x)
    
    # Create models
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    
    return autoencoder, encoder

def train_autoencoder(autoencoder, filt_ripples, epochs=50, batch_size=32):
    """
    Train the autoencoder model.
    
    Args:
        autoencoder: The autoencoder model to train
        filt_ripples: Input data with shape (num_samples, 127, 1)
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        history: Training history
    """
    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Train the model
    history = autoencoder.fit(
        filt_ripples, filt_ripples,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.2,
        verbose=1
    )
    
    return history

def extract_bottleneck_features(encoder, filt_ripples):
    """
    Extract features from the bottleneck layer.
    
    Args:
        encoder: The encoder model
        filt_ripples: Input data with shape (num_samples, 127, 1)
        
    Returns:
        bottleneck_features: Extracted features
    """
    bottleneck_features = encoder.predict(filt_ripples)
    # bottleneck_features shape will be (num_samples, 16, encoding_dim)
    # Flatten to (num_samples, 16*encoding_dim) for dimensionality reduction
    bottleneck_features_flat = bottleneck_features.reshape(bottleneck_features.shape[0], -1)
    
    return bottleneck_features_flat

def visualize_reconstruction(autoencoder, filt_ripples, num_examples=5):
    """
    Visualize original and reconstructed waveforms.
    
    Args:
        autoencoder: The trained autoencoder model
        filt_ripples: Input data with shape (num_samples, 127, 1)
        num_examples: Number of examples to visualize
    """
    # Get reconstructions
    reconstructions = autoencoder.predict(filt_ripples[:num_examples])
    
    # Plot
    plt.figure(figsize=(15, 10))
    for i in range(num_examples):
        # Original
        plt.subplot(num_examples, 2, 2*i + 1)
        plt.plot(filt_ripples[i, :, 0])
        plt.title(f'Original #{i+1}')
        
        # Reconstruction
        plt.subplot(num_examples, 2, 2*i + 2)
        plt.plot(reconstructions[i, :, 0])
        plt.title(f'Reconstruction #{i+1}')
    
    plt.tight_layout()
    plt.show()


# Build the autoencoder
autoencoder, encoder = build_convolutional_autoencoder(input_shape=(127, 1), encoding_dim=16)

# Print model summary
autoencoder.summary()

# Train the autoencoder
history = train_autoencoder(autoencoder, filt_ripples, epochs=50)


# Save the full autoencoder
autoencoder.save('autoencoder_model.keras')

# Save just the encoder
encoder.save('encoder_model.keras')


# Extract bottleneck features for dimensionality reduction
bottleneck_features = extract_bottleneck_features(encoder, filt_ripples)
print(f"Bottleneck features shape: {bottleneck_features.shape}")

# Visualize some examples
visualize_reconstruction(autoencoder, filt_ripples)



embedding = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=3, metric='euclidean')
embedding.fit(bottleneck_features)

# Save embedding matrix
save_path = '/home/eloy/Code/DeepSup/Analysis/DeepSupDB/Original/'
emb = embedding.embedding_
dic = {}
for dim in range(embedding.embedding_.shape[1]):
	dic['x' + str(dim+1)] = emb[:, dim]
real_df = pd.DataFrame(dic)
real_df.to_csv(save_path + 'emb_autoencoder_filtRipples.csv')


