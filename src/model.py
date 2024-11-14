import os
import gc
import json
import tensorflow as tf
from transformers import TFSegformerModel, TFSegformerDecodeHead

class DualSegformerClassifierModel(tf.keras.models.Model):
    def __init__(self, 
                 pretrained_path = 'nvidia/mit-b0', 
                 input_size = (512, 512, 9),
                 output_size = (256, 256),
                 classifier_depth = 128,
                 num_classif_labels  = 2,
                 num_seg_labels = 2,
                 label2id = None,
                 id2label = None,
                 *args, 
                 **kwargs
                ):
        
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.pretrained_path = pretrained_path
        self.input_size = input_size
        self.classifier_depth = classifier_depth
        if num_classif_labels == 2:
            self.num_classif_labels = 1
            self.classif_activation = 'sigmoid'
        else:
            self.num_classif_labels = num_classif_labels
            self.classif_activation = 'softmax'
        self.num_seg_labels = num_seg_labels
        self.label2id = label2id
        self.id2label = id2label

        # build encoder
        self.encoder = TFSegformerModel.from_pretrained(
            pretrained_path
        )

        # embed the year variable to concatenate with the segformer embeddings
        w1 = self.input_size[0] // 32
        w2 = self.input_size[1] // 32
        w3 = self.encoder.config.hidden_sizes[-1]
        self.embedder = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape = (1)),
            tf.keras.layers.Dense(units = w1 * w2 * w3, name = 'embedder_proj'),
            tf.keras.layers.Reshape([w3, w1, w2], name = 'embedder_reshaper')
        ], name = 'embedder')
        
        # The images have more than 3 channels
        # Preprocess with a CNN to reduce to 3 channels before passing to Segformer
        self.preprocessor = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape = self.input_size),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                64, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', 
                name = 'image_prepocessor', data_format = 'channels_last'),
            tf.keras.layers.Conv2D(
                128, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', 
                name = 'image_prepocessor_2', data_format = 'channels_last'),
            tf.keras.layers.Conv2D(
                3, kernel_size = 1, strides = 1, padding = 'same', activation = None, 
                name = 'image_prepocessor_3', data_format = 'channels_last'),
            tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm = [0, 3, 1, 2]), name = 'transpose')
        ])
        
        # layer to add image and year embeddings
        self.add = tf.keras.layers.Add(name = 'latent_add') 
        
        # segformer decode head
        self.decode_head = TFSegformerDecodeHead.from_pretrained(
            pretrained_path,
            num_labels = self.num_seg_labels,
            id2label = self.id2label,
            label2id = self.label2id
        )
        self.softmax = tf.keras.layers.Activation('softmax') # FUTURE: implement softmax with temperature
        self.segmentation_head = tf.keras.layers.Lambda(lambda x : tf.image.resize(x, size = (self.output_size[0], self.output_size[1]), method = 'bilinear'))

        # Localization classifier head
        self.classifier_head = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                self.encoder.config.hidden_sizes[0], kernel_size = 1, strides = 1, padding = 'same', activation = 'relu', 
                name = 'classifier_reducer', data_format = 'channels_first'
            ),
            tf.keras.layers.Flatten(name = 'classifier_flatten'),
            tf.keras.layers.Dense(self.encoder.config.hidden_sizes[0], name = 'classifier_projection', activation = 'relu'),
            tf.keras.layers.Dense(self.num_classif_labels, activation = self.classif_activation, name = 'classifier_output')
        ], name = 'classifier_head')
    
    def call(self, inputs, training=False):
        # Get inputs
        yr = inputs[0]
        feature = inputs[1]

        # Preprocess the feature input
        x = self.preprocessor(feature, training = training)

        # Encode the image features using Segformer
        x = self.encoder(x, output_hidden_states = True, training=training).hidden_states

        # Embed the year variable and concatenate with image features
        yr = self.embedder(yr, training = training)
        last_hidden_state = self.add([x[-1], yr], training = training)

        # Classification output
        classif = self.classifier_head(last_hidden_state, training = training)

        # Segmentation output
        segmentation = self.decode_head([*x[:-1], last_hidden_state], training = training)
        segmentation = self.softmax(segmentation)
        segmentation = self.segmentation_head(segmentation)

        return (classif, segmentation)

    def summary(self, *args, **kwargs):
        self.encoder.summary(*args, **kwargs)
        self.preprocessor.summary(*args, **kwargs)
        self.embedder.summary(*args, **kwargs)
        self.decode_head.summary(*args, **kwargs)
        self.classifier_head.summary(*args, **kwargs)
        
    @classmethod
    def from_pretrained(cls, path):
        # Ensure the directory exists
        if not os.path.isdir(path):
            raise ValueError(f"Model path {path} does not exist.")
        
        # Load model configuration from saved JSON or YAML
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize the model with the loaded configuration
        model = cls(**config)
        
        # Load the weights for each component (excluding segmentation_head)
        model.encoder.load_weights(os.path.join(path, "encoder_weights"))
        model.classifier_head.load_weights(os.path.join(path, "classifier_head_weights"))
        model.embedder.load_weights(os.path.join(path, "embedder_weights"))
        model.preprocessor.load_weights(os.path.join(path, "preprocessor_weights"))
        
        return model

    def save_model(self, path):
        # Create the directory to store model weights and configuration if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Save the model's configuration (to ensure reproducibility)
        config = {
            'pretrained_path': self.pretrained_path,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'classifier_depth': self.classifier_depth,
            'num_classif_labels': self.num_classif_labels,
            'num_seg_labels': self.num_seg_labels,
            'label2id': self.label2id,
            'id2label': self.id2label
        }
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        # Save the weights of each model component (excluding segmentation_head)
        self.encoder.save_weights(os.path.join(path, "encoder_weights"))
        self.classifier_head.save_weights(os.path.join(path, "classifier_head_weights"))
        self.embedder.save_weights(os.path.join(path, "embedder_weights"))
        self.preprocessor.save_weights(os.path.join(path, "preprocessor_weights"))

    def train_step(self, data):
        x, (y_classif, y_seg) = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            classif, segmentation = self(x, training=True)
            
            # Compute loss for both tasks
            total_loss = self.compiled_loss((y_classif, y_seg), (classif, segmentation))
        
        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (for accuracy, loss, etc.)
        self.compiled_metrics.update_state((y_classif, y_seg), (classif, segmentation))
        
        # Return a dictionary of performance
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, (y_classif, y_seg) = data
        
        # Forward pass
        classif, segmentation = self(x, training=False)
        
        # Compute loss for both tasks
        # Total loss
        total_loss = self.compiled_loss((y_classif, y_seg), (classif, segmentation))

        # Update metrics (for accuracy, loss, etc.)
        self.compiled_metrics.update_state((y_classif, y_seg), (classif, segmentation))

        # Return a dictionary of performance
        return {m.name: m.result() for m in self.metrics}
    
class ClearMemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Clear the backend session
        tf.keras.backend.clear_session()
        
        # Run garbage collection
        gc.collect()