from args_parser_opts import get_args
import os
import math
import shutil
from pathlib import Path

dataset_folder = "original_dataset"
non_iid_dataset_folder = "non_iid_dataset"

full_dataset = {}
total_black_samples = 0
total_blue_samples = 0
total_green_samples = 0


def get_all_files():

    class_dict = {"blue": [], "black": [], "green": []}

    for class_ in class_dict.keys():
        for path, _, files in os.walk(os.path.join("./", dataset_folder, class_)):
            for name in files:
                if name.endswith(".png") or name.endswith(".jpeg") or name.endswith(".jpg"):
                    class_dict[class_].append(os.path.join(path, name))

    return class_dict


def get_samples_from_class(class_name, num_samples_for_class):

    bin_dict_for_class = {class_name: []}

    for sample in full_dataset[class_name]:
        bin_dict_for_class[class_name].append(sample)

        if len(bin_dict_for_class[class_name]) == num_samples_for_class:
            break

    for assigned_sample in bin_dict_for_class[class_name]:
        if assigned_sample in full_dataset[class_name]:
            full_dataset[class_name].remove(assigned_sample)

    return bin_dict_for_class[class_name]


def get_samples(percents, name_bin):

    print("Assigning samples to {} bin dataset:".format(name_bin))

    num_black_samples = math.floor(percents[0]*total_black_samples)
    num_blue_samples = math.floor(percents[1]*total_blue_samples)
    num_green_samples = math.floor(percents[2]*total_green_samples)

    bin_dict = {"blue": [], "black": [], "green": []}

    bin_dict["black"] = get_samples_from_class("black", num_black_samples)
    bin_dict["blue"] = get_samples_from_class("blue", num_blue_samples)
    bin_dict["green"] = get_samples_from_class("green", num_green_samples)

    print("     Assigned {} black samples".format(len(bin_dict["black"])))
    print("     Assigned {} blue samples".format(len(bin_dict["blue"])))
    print("     Assigned {} green samples".format(len(bin_dict["green"])))

    print("\n")

    return bin_dict


if __name__ == '__main__':

    args = vars(get_args())

    full_dataset = get_all_files()

    total_black_samples = len(full_dataset["black"])
    total_blue_samples = len(full_dataset["blue"])
    total_green_samples = len(full_dataset["green"])

    total_samples = total_black_samples + total_blue_samples + total_green_samples

    print("Total samples: {}\n".format(total_samples))

    print("Black samples: {}".format(total_black_samples))
    print("Blue samples: {}".format(total_blue_samples))
    print("Green samples: {}\n".format(total_green_samples))

    # Assigning black bin samples
    black_bin_samples = get_samples(args["black_bin_percents"], "black")

    # Assigning blue bin samples
    blue_bin_samples = get_samples(args["blue_bin_percents"], "blue")

    # Assigning green bin samples
    green_bin_samples = get_samples(args["green_bin_percents"], "green")

    bins = ["green_bin", "blue_bin", "black_bin"]
    classes = ["black", "blue", "green"]
    dict_of_samples = {"green": green_bin_samples,
                       "black": black_bin_samples, "blue": blue_bin_samples}

    for bin in bins:

        bin_colour = bin.split("_")[0]

        print("Bin colour: {}".format(bin_colour))

        for _class in classes:

            dest_path = os.path.join(
                non_iid_dataset_folder, bin, _class)
            Path(dest_path).mkdir(parents=True, exist_ok=True)

            samples_for_a_class_for_a_bin = dict_of_samples[bin_colour][_class]

            print("    Copying {} {} samples to the {} bin...".format(
                len(samples_for_a_class_for_a_bin), _class, bin_colour))

            for sample in samples_for_a_class_for_a_bin:
                try:
                    shutil.copy(sample, dest_path)
                except IOError as e:
                    print("Unable to copy file: {}".format(e))
        print("\n")
