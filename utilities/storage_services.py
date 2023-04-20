import os


def rename_data_set(input_dir):
    count = 1
    prefix_file_name = "original-"
    paths = ["train", "test", "valid"]
    for data_dir in paths:
        images_path = os.path.join(input_dir, data_dir, "images")
        labels_path = os.path.join(input_dir, data_dir, "labels")
        for file in os.listdir(images_path):
            img_path = os.path.join(images_path, file)
            label_path = os.path.join(labels_path, file.replace(".jpg", ".txt"))
            rename_img_path = os.path.join(images_path, f"{prefix_file_name}{count}.jpg")
            rename_label_path = os.path.join(labels_path, f"{prefix_file_name}{count}.txt")
            count += 1
            os.rename(img_path, rename_img_path)
            os.rename(label_path, rename_label_path)


def ensure_directory_exists(path):
    """
    Ensure that the directory at `path` exists, creating it if necessary.
    """
    if not os.path.exists(path):
        os.makedirs(path)
