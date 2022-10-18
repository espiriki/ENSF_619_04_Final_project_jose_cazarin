import argparse

# black class, blue class, green class
black_bin_array = [0.75, 0.20, 0.07]

# black class, blue class, green class
blue_bin_array = [0.15, 0.75, 0.08]

# black class, blue class, green class
green_bin_array = [0.1, 0.05, 0.85]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--blue-bin-percents', nargs='+', type=float, default=blue_bin_array,
                        help="%% of samples of each class in the blue bin dataset")

    parser.add_argument('--green-bin-percents', nargs='+', type=float, default=green_bin_array,
                        help="%% of samples of each class in the green bin dataset")

    parser.add_argument('--black-bin-percents', nargs='+', type=float, default=black_bin_array,
                        help="%% of samples of each class in the black bin dataset")

    return parser.parse_args()
