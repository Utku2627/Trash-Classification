# imports
import os
import shutil
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import (load_img)

# view categories
categories: List[str] = os.listdir(r"C:\TrashClassification\.data")
print('The categories are: ', categories)


# count the number of images in each category
image_paths = {}
i = 0
for cat in categories:
    image_paths[cat] = os.listdir(os.path.join(r"C:\TrashClassification\.data", cat))
    print('There are', len(os.listdir(os.path.join(r"C:\TrashClassification\.data", cat))), cat, 'images')
    i = i + len(os.listdir(os.path.join(r"C:\TrashClassification\.data", cat)))
print("TOTAL", i, "PHOTOS IN THE DATASET")


# inspect random images
fig, axs = plt.subplots(1, 4, figsize=(16, 4))

for ax in axs:
    # choose a random category
    cat = np.random.choice(categories)

    # choose a random image in that category
    file_name = np.random.choice(os.listdir(os.path.join(r"C:\TrashClassification\.data", cat)))

    # load and display the image
    img = load_img(os.path.join(r"C:\TrashClassification\.data", cat, file_name))
    ax.imshow(img)
    ax.set_title(file_name)
plt.show()


# set proportion of each class to hold for validation and testing.
val_portion = 0.13
test_portion = 0.17

split_base = r"C:\TrashClassification\splitted_data"


# remove the folder if it exists and rerun the split
if os.path.isdir(split_base):
    shutil.rmtree(split_base)

os.mkdir(split_base)


# store the path of each folder and create directories
train_folder = os.path.join(split_base, 'train')
os.mkdir(train_folder)

val_folder = os.path.join(split_base, 'validation')
os.mkdir(val_folder)

test_folder = os.path.join(split_base, 'test')
os.mkdir(test_folder)


# Split the data into training, validation, and split folders
for cat in categories:
    # get total number of files in this category
    num_files = len(image_paths[cat])

    # randomize the file order for this category
    np.random.shuffle(image_paths[cat])

    # set split boundaries for validation and test
    validation_boundary = int(np.floor(num_files * val_portion))
    test_boundary = int(np.floor(num_files * test_portion) + validation_boundary)

    # store boundaries
    labels = {'validation': image_paths[cat][:validation_boundary],
              'test': image_paths[cat][validation_boundary:test_boundary],
              'train': image_paths[cat][test_boundary:]}

    # loop through boundaries and move copy files
    for label, files in labels.items():
        # create folder
        os.mkdir(os.path.join(split_base, label, cat))

        # copy files to the correct directory
        for file in files:
            shutil.copyfile(os.path.join(r"C:\TrashClassification\.data", cat, file),
                            os.path.join(split_base, label, cat, file))


# Check number of files in each folder
def count_files(folder):
    paths = [path for path, subdirs, dossier in os.walk(folder) if path != folder]
    num = 0
    for path in paths:
        num += len(os.listdir(path))
    return num


# count_files is a custom function in utils.py
num_train = count_files(train_folder)
num_val = count_files(val_folder)
num_test = count_files(test_folder)

print('Total training files:', num_train)
print('Total validation files:', num_val)
print('Total testing files:', num_test)


# The folder and number of files in it for each class in the training set
folders = [(f.path, len(os.listdir(f.path))) for f in os.scandir(train_folder) if f.is_dir()]


# sort the folders in descending order by number of files
folders = sorted(folders, key=lambda x: x[1], reverse=True)

train_dir = r"C:\\TrashClassification\\splitted_data\\train"
validation_dir = r"C:\\TrashClassification\\splitted_data\\validation"
test_dir = r"C:\\TrashClassification\\splitted_data\\test"

classes = os.listdir(train_dir)

# Move 15 random images from train to either validation or test
for _ in range(15):
    # Randomly select a class directory
    class_name = np.random.choice(classes)
    from_dir = os.path.join(train_dir, class_name)

    # List all files in the selected class directory
    files = os.listdir(from_dir)
    if not files:
        continue  # Skip if the directory is empty

    # Randomly select a file from this class directory
    file = np.random.choice(files)

    # Randomly choose the target directory (validation or test)
    to_dir = np.random.choice([validation_dir, test_dir])
    target_dir = os.path.join(to_dir, class_name)

    # Create the target class directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Move the file
    print(f'Moving {file} from {from_dir} to {target_dir}')
    shutil.move(os.path.join(from_dir, file), os.path.join(target_dir, file))

# Print the final values
num_train = count_files(train_folder)
num_val = count_files(val_folder)
num_test = count_files(test_folder)

print('Total training files:', num_train)
print('Total validation files:', num_val)
print('Total testing files:', num_test)
