from utilities.augment_services import augment_dataset
from utilities.storage_services import rename_data_set


def main():
    input_dir = "chromo.v6i.yolov8"
    output_dir = "chromo.v6i.yolov8.augmented"
    # rename_data_set(input_dir)
    augment_dataset(input_dir, output_dir, ["flip", "brightness", "noise", "crop"])


if __name__ == "__main__":
    main()
