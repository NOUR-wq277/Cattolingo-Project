"""
Model architecture for Cat Classification using VGG16
"""
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
from config import INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE_PHASE1, LEARNING_RATE_PHASE2


def create_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Create VGG16-based model for cat classification
    
    Args:
        input_shape: Tuple of (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained VGG16 model without top layers
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers initially
    for layer in base_model.layers:
        layer.trainable = False
    
    # Build custom classification head
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create final model
    model = models.Model(inputs=base_model.input, outputs=output)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE_PHASE1),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def unfreeze_last_layers(model, num_layers=4, learning_rate=LEARNING_RATE_PHASE2):
    """
    Unfreeze last N layers of the base model for fine-tuning
    
    Args:
        model: Keras model
        num_layers: Number of layers to unfreeze from the end
        learning_rate: Learning rate for fine-tuning
        
    Returns:
        Model with unfrozen layers
    """
    # Get the base model (VGG16 part)
    base_model = model.layers[1] if len(model.layers) > 1 else model
    
    # Unfreeze last N layers
    for layer in base_model.layers[-num_layers:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
