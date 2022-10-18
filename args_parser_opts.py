import argparse

# Note that the values in the first element of each bin array sum to 1.0
# I did this to use all the samples for a given class
# In this way the samples for the class "black" will be distributed like this:
# black bin will receive 75% of the samples of "black" class
# blue bin will receive 15% of the samples of "black" class
# green bin will receive 10% of the samples of "black" class
# Totalling 100% of the samples of the "black" class
#
# Same applies to the samples of the classes blue and green for the other bins

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
