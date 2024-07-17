import os
from keras_preprocessing.image import ImageDataGenerator


# Reading .npy files
def load_data(base_dir, folder_name, image_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1. / 255)
    data_path = os.path.join(base_dir, folder_name)
    data_flow = datagen.flow_from_directory(
        data_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    return data_flow


def count_files(folder):
    """Counts all files in subdirectories a give directory

    Parameters:
    -----------
    folder: The directory to containing the files to count

    Returns:
    --------
    count:  int. The number of files not include folders, but including all
    subdirectories in the given path.
    """
    # get subdirectories
    paths = [path for path, subdirs, files in os.walk(folder) if path != folder]
    num = 0

    # recurse subdirectories and count files
    for path in paths:
        num += len(os.listdir(path))
    return num
