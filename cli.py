import argparse

if __name__ == "__main__":
    print("Can't run cli.py on its own...")
    exit(1)

__parser__ = argparse.ArgumentParser(
    prog="MtG-Scan-Extractor",
    description="Extracts MtG cards from scans",
)
__parser__.add_argument(
    "-i", "--input", help="Path to input, either a single scan or a folder.", required=True
)
__parser__.add_argument(
    "-o", "--output", help="Path to a folder to write outputs to.", required=True
)
__parser__.add_argument(
    "-d", "--downsample", type=int, default=1, help="Factor by which to downsample images."
)
__parser__.add_argument(
    "-dd", "--display_downsample", type=int, default=5, help="Factor by which to downsample images before showing them to the user."
)
__parser__.add_argument(
    "-c", "--center", default=False, action="store_true", help="Center the frame rather than the physical card."
)
__parser__.add_argument(
    "-v", "--verbose", action="store_true", help="Print out verbose info."
)

__args__ = __parser__.parse_args()

program = __parser__.prog
input = __args__.input
output = __args__.output
downsample = __args__.downsample
display_downsample = __args__.display_downsample
center = __args__.center


def no_print(*args, **kwargs):
    pass

verbose = __args__.verbose
print_verbose = print if verbose else no_print
