# Load library
from keras.preprocessing.image import ImageDataGenerator
# Create image augmentation
augmentation = ImageDataGenerator(featurewise_center=True, # Apply ZCA whitening
                                  zoom_range=0.3, # Randomly zoom in on images
                                  width_shift_range=0.2, # Randomly shift images
                                  horizontal_flip=True, # Randomly flip images
                                  rotation_range=90) # Randomly rotate
# Process all images from the directory 'raw/images'
augment_images = augmentation.flow_from_directory("raw/images", # Image folder
                                                  batch_size=32, # Batch size
                                                  class_mode="binary", # Classes
                                                  save_to_dir="processed/images")
