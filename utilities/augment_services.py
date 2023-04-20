import os
import random
import cv2
import numpy as np
from utilities.storage_services import ensure_directory_exists


def flip_image(image, label):
    # Flip the image horizontally
    flipped_image = cv2.flip(image, 1)

    # Update the label coordinates accordingly
    flipped_label = label.copy()
    # for bbox in flipped_label:
    #     bbox[1] = 1 - bbox[1]
    flipped_label[:, 1] = 1 - flipped_label[:, 1]

    return flipped_image, flipped_label


def random_crop(image, labels, crop_width=200, crop_height=200):
    img_height, img_width = image.shape[:2]

    # Choose a random crop position
    crop_x = random.randint(0, img_width - crop_width)
    crop_y = random.randint(0, img_height - crop_height)

    # Crop the image
    cropped_image = image[crop_y: crop_y + crop_height, crop_x: crop_x + crop_width]

    # Update the labels
    cropped_labels = []
    for label in labels:
        x_center = label[1]
        y_center = label[2]
        w = label[3]
        h = label[4]

        # Convert bounding box coordinates to pixel coordinates
        x1 = int((x_center - w / 2) * img_width)
        y1 = int((y_center - h / 2) * img_height)
        x2 = int((x_center + w / 2) * img_width)
        y2 = int((y_center + h / 2) * img_height)

        # Apply the crop to the bounding box coordinates
        x1 -= crop_x
        y1 -= crop_y
        x2 -= crop_x
        y2 -= crop_y

        # Discard boxes that are completely outside the crop
        if x2 < 0 or y2 < 0 or x1 > crop_width or y1 > crop_height:
            continue

        # Clip the boxes that are partially outside the crop
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(crop_width, x2)
        y2 = min(crop_height, y2)

        # Convert bounding box coordinates back to YOLO format
        x_center = (x1 + x2) / (2 * crop_width)
        y_center = (y1 + y2) / (2 * crop_height)
        w = (x2 - x1) / crop_width
        h = (y2 - y1) / crop_height

        # Add the updated label to the list
        cropped_labels.append([label[0], x_center, y_center, w, h])

    return cropped_image, np.array(cropped_labels)


def add_gaussian_noise(image, label, mean=0, std=10):
    # Generate random Gaussian noise and add it to the image
    noise = np.random.normal(mean, std, size=image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)

    return noisy_image, label


def adjust_brightness(image, label, brightness_factor):
    # Adjust the brightness of the image
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * brightness_factor, 0, 255)
    bright_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return bright_image, label


def augment_image(image, label, augment):
    if augment == "flip":
        return flip_image(image, label)

    if augment == "crop":
        return random_crop(image, label)

    if augment == "noise":
        return add_gaussian_noise(image, label)

    if augment == "brightness":
        return adjust_brightness(image, label, 1.5)


def augment_dataset(input_dir: str, output_dir: str, augmentations: []):
    # Loop through all image and label files in the input directory
    paths = ["train", "test", "valid"]
    for data_dir in paths:
        images_path = os.path.join(input_dir, data_dir, "images")
        labels_path = os.path.join(input_dir, data_dir, "labels")
        output_images_dir = os.path.join(output_dir, data_dir, "images")
        output_labels_dir = os.path.join(output_dir, data_dir, "labels")
        ensure_directory_exists(output_images_dir)
        ensure_directory_exists(output_labels_dir)
        for file in os.listdir(images_path):
            img_path = os.path.join(images_path, file)
            label_path = os.path.join(labels_path, file.replace(".jpg", ".txt"))
            img = cv2.imread(img_path)
            label = []
            with open(label_path, "r") as f:
                temp = f.read().strip().split("\n")
                if temp[0] != '':
                    for line in temp:
                        class_id, x, y, w, h = map(float, line.strip().split(' '))
                        label.append([class_id, x, y, w, h])
            if len(label) == 0:
                continue
            # Apply all augmentations to the image and label
            img_aug, label_aug = img, np.array(label)
            for augmentation in augmentations:
                new_img_aug, new_label_aug = augment_image(
                    img_aug, label_aug, augmentation)

                # Save the augmented image and label to the output directory
                out_img_path = os.path.join(
                    output_images_dir, f"{file[:-4]}_{augmentation}_augmented.jpg")
                out_label_path = os.path.join(
                    output_labels_dir, f"{file[:-4]}_{augmentation}_augmented.txt")
                cv2.imwrite(out_img_path, new_img_aug)
                list_label = new_label_aug.tolist()
                with open(out_label_path, "w") as f:
                    for label in list_label:
                        # Convert the label array to a string with space-separated values
                        label_str = " ".join(str(x) for x in label)
                        # Write the label string to the file
                        f.write(label_str + "\n")
            for i in range(len(augmentations)):
                i_augmentation = augmentations[i]
                augmentations_name = f"_{i_augmentation}"
                new_img_aug, new_label_aug = augment_image(
                    img_aug, label_aug, i_augmentation)
                for j in range(i + 1, len(augmentations)):
                    j_augmentation = augmentations[j]
                    augmentations_name += f"_{j_augmentation}"
                    new_img_aug, new_label_aug = augment_image(
                        new_img_aug, new_label_aug, j_augmentation)

                    # Save the augmented image and label to the output directory
                    out_img_path = os.path.join(
                        output_images_dir, f"{file[:-4]}_{augmentations_name}_augmented.jpg")
                    out_label_path = os.path.join(
                        output_labels_dir, f"{file[:-4]}_{augmentations_name}_augmented.txt")
                    cv2.imwrite(out_img_path, new_img_aug)
                    list_label = new_label_aug.tolist()
                    with open(out_label_path, "w") as f:
                        for label in list_label:
                            # Convert the label array to a string with space-separated values
                            label_str = " ".join(str(x) for x in label)
                            # Write the label string to the file
                            f.write(label_str + "\n")
