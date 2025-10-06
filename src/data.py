"""
Data loading and preprocessing utilities
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import (
    BATCH_SIZE, INPUT_SHAPE,
    ROTATION_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE,
    ZOOM_RANGE, HORIZONTAL_FLIP, FILL_MODE,
    TRAIN_DATA_DIR, VAL_DATA_DIR
)


def create_data_generators(
    train_dir=TRAIN_DATA_DIR,
    val_dir=VAL_DATA_DIR,
    target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
    batch_size=BATCH_SIZE
):
    """
    Create training and validation data generators with augmentation
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        target_size: Target image size (height, width)
        batch_size: Batch size for training
        
    Returns:
        Tuple of (train_generator, val_generator)
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        zoom_range=ZOOM_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        fill_mode=FILL_MODE
    )
    
    # Validation data generator (only rescaling)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, val_generator


def get_class_names(generator):
    """
    Get class names from data generator
    
    Args:
        generator: Keras ImageDataGenerator
        
    Returns:
        List of class names
    """
    class_indices = generator.class_indices
    return {v: k for k, v in class_indices.items()}
