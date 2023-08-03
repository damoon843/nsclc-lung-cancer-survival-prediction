import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from sklearn.model_selection import train_test_split
import keras.backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from lifelines.utils import concordance_index
#from sksurv.metrics import integrated_brier_score


### CONSTANTS ### 

SKIPPED = [58] # Scans that cannot be used
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
BATCH_SIZE = 20

STUDY_DURATION = 365 * 3
NUM_CT_SCANS = 422
TRAIN_SIZE = 0.80 
#VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

### UTILITY FUNCTIONS ### 

def load_data():
    """
    Loads preprocessed dataset with labels.
    """
    lungs = np.load(os.path.join(os.getcwd(), 'preprocessed_lungs.npy'), allow_pickle=True)
    patient_df = pd.read_csv(os.path.join(os.getcwd(), 'NSCLC Radiomics Lung1.clinical-version3-Oct 2019.csv'))
    # Remove patient data w/o ORIGINAL scan
    for i in SKIPPED: 
        df = patient_df.drop(i, axis=0)
        
    # Extract survival times and dead status events
    times = df['Survival.time'].to_numpy()
    events = np.round(df['deadstatus.event'].to_numpy())
    
    # Add channel to lungs
    lungs = np.expand_dims(lungs, axis=4)
        
    return lungs, times, events
    

def train_val_test_split(X, y, events):
    """ 
    Split dataset into train, validation, and test.
    """
    
    X_train, X_test, y_train, y_test, e_train, e_test = train_test_split(X, y, events, test_size=TEST_SIZE, random_state=1)
    X_train, X_val, y_train, y_val, e_train, e_val = train_test_split(X_train, y_train, e_train, test_size=0.17467, random_state=1) # 0.17647 x 0.85 = 0.15
    
    return X_train, y_train, e_train, X_val, y_val, e_val, X_test, y_test, e_test    


def get_outputs(inputs, depth=64, height=128, width=128):
    """
    Builds 3D CNN model.
    
    Annoyance with batch normalization: https://github.com/tensorflow/tfjs/issues/5843.
    Workaround for batch normalization with 5D tensors: https://github.com/tensorflow/tensorflow/issues/5694.
    Ways to deal with small dataset: data augmentation or transfer learning
    """
    #inputs = keras.Input((depth, width, height, 1), batch_size=BATCH_SIZE)
    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # height = width = x.shape[2]
    # x = tf.reshape(x, [BATCH_SIZE, x.shape[1], height * width, x.shape[4]])
    # x = layers.BatchNormalization()(x)
    # x = tf.reshape(x, [BATCH_SIZE, x.shape[1], height, width, x.shape[3]])
    
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    
    # x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPool3D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.GlobalAveragePooling3D()(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    # x = layers.Dense(units=256, activation="relu")(x)
    # x = layers.Dropout(0.3)(x)

    # NOTE: output is log-risk function
    outputs = layers.Dense(units=1, activation="linear", kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.L2(0.01))(x)
    
    # # Define the model.
    # return keras.Model(inputs, outputs, name="3dcnn")  
    
    return outputs

def negative_log_likelihood_loss(y_true_events, y_pred):
    """Custom loss function."""  
    print(y_true_events.shape)
    print(y_true_events)
    y_true = y_true_events[:, 0]
    events = y_true_events[:, 1]
    
    hazard_ratio = K.exp(y_pred)
    log_risk = K.log(K.cumsum(hazard_ratio))
    uncensored_likelihood = y_pred - log_risk
    censored_likelihood = uncensored_likelihood * events
    neg_likelihood = -K.sum(censored_likelihood)
    
    return neg_likelihood  

### HELPER CLASSES ### 

class DataGenerator(keras.utils.Sequence):
    """
    Dataset wrapper class to generate batches.
    
    Code adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.
    """
    def __init__(self, x_set, y_set, e_set, batch_size=BATCH_SIZE, dim=(64, 128, 128), n_channels=1, shuffle=True):
        self.x, self.y, self.e = x_set, y_set, e_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])
        self.shuffle = shuffle
        self.dim = dim
        self.n_channels = n_channels

    def __len__(self):
        """Return number of batches per epoch."""
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, idx):
        """Generate one batch."""
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        # Generate data (note how __data_generation returns (y,events) in a list)
        X, y_e = self.__data_generation(inds)
                
        return X, y_e
    
    def __data_generation(self, target_ids):
        """Helper to generate one batch of data.""" # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_X = np.empty((self.batch_size, *self.dim, self.n_channels))
        batch_y = np.empty((self.batch_size), dtype=np.float32)
        batch_e = np.empty((self.batch_size), dtype=np.float32)

        # Generate data
        for i, ID in enumerate(target_ids):
            batch_X[i] = self.x[i]
            batch_y[i] = self.y[i]
            batch_e[i] = self.e[i]
            
        return batch_X, [batch_y, batch_e]

    def on_epoch_end(self):
        """Update indices after every epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)

class cnn3d(tf.keras.models.Model):
    
    def train_step(self, data):   
        # https://stackoverflow.com/questions/69315586/when-are-model-call-and-train-step-called
        # TODO: 
        x, y = data
        
        print(x)
        print(y)
        
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x) 
            # Compute the loss value
            loss = self.compiled_loss(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return loss           


### MODEL SETUP ### 
# TODO: use cross validation with k=10, average CI from 10 test sets
# Load and split dataset
X, y, events = load_data()
X_train, y_train, e_train, X_val, y_val, e_val, X_test, y_test, e_test = train_val_test_split(X, y, events)
print(f"SIZE OF TRAIN SET: {y_train.shape}")
print(f"SIZE OF VAL SET: {y_val.shape}")
print(f"SIZE OF TEST SET: {y_test.shape}")

# Instantiate generators
training_generator = DataGenerator(X_train, y_train, e_train)
validation_generator = DataGenerator(X_val, y_val, e_val)
test_generator = DataGenerator(X_test, y_test, e_test)

# Load and fit model
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="loss", mode="min", patience=10)
#model = get_model()
inputs = keras.Input((128, 64, 64, 1), batch_size=BATCH_SIZE)
outputs = get_outputs(inputs=inputs)
model = cnn3d(inputs, outputs)

### MODEL TRAINING ### 
model.compile(
    loss=negative_log_likelihood_loss,
    optimizer=keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE),
    metrics=[],
)

history = model.fit(
    x=training_generator,
    epochs=NUM_EPOCHS,
    callbacks=[early_stopping_cb],
    validation_data=validation_generator,
)

# Plot train and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

print(train_loss)
print(val_loss)

plt.plot(train_loss, label="train")
plt.plot(val_loss, label="validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Loss Curve")
plt.legend()
plt.savefig(os.path.join(os.getcwd(), "loss_curve.png"))
#plt.show()

# Generate predictions for survival times
preds = model.predict(x=test_generator)

# Evaluate model
""" 
Prediction of risk of death: Brier and CI (time-dependent ROC)
Prediction of Survival times: CI (time-dependent ROC)

Cannot evaluate Brier score b/c CNN does not rely on proportional hazards assumption: https://scikit-survival.readthedocs.io/en/stable/user_guide/understanding_predictions.html#predictions
"""
ci = concordance_index(y_test, -np.exp(preds), e_test) # Reason for negative preds: https://stats.stackexchange.com/questions/352183/use-median-survival-time-to-calculate-cph-c-statistic/352435#352435
# survival_train = zip(e_train, y_train)
# survival_test = zip(e_test, y_test)
# times = np.arange(1, STUDY_DURATION+1, 5)
#ibs = integrated_brier_score(survival_train, survival_test,  )

print(ci)










