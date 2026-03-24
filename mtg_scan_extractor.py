# built-in
import os
import itertools
import functools
from collections.abc import Iterable
from pathlib import Path

# 3rd-party
import cv2
import numpy as np
from pypdf import PdfReader

# internal
import cli
from cli import print_verbose
from scan_types import *


BoundsLines = tuple[tuple[Line, Line], tuple[Line, Line]]

DISPLAY_DOWNSAMPLE = 1

CARD_WIDTH = 2.48
CARD_HEIGHT = 3.46

BORDER_HEIGHT = 0.13916666666666666666666666666667
BORDER_WIDTH = 0.11791666666666666666666666666667


def get_files(folder: Path, extensions: Iterable[str]):
    if not folder.is_dir():
        return None

    def is_image(path: Path):
        return path.is_file() and path.suffix in extensions

    images = filter(is_image, folder.glob("*"))
    return list(sorted(images))


def get_pdfs(folder: Path):
    return get_files(folder, (".pdf"))


def get_images(folder: Path):
    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".tiff",
        ".tif",
        ".bmp",
        ".gif",
    )
    return get_files(folder, image_extensions)


def extract_pdf_images(pdf_path: Path, interim: Path):
    extracted_images = []

    reader = PdfReader(pdf_path)
    for page in reader.pages:
        for image in page.images:
            image_path = interim / Path(f"{pdf_path.stem}_{image.name}")
            extracted_images.append(image_path)
            if not image_path.exists():
                with open(image_path, "wb") as fp:
                    fp.write(image.data)

    return extracted_images


def downsample_image(image: cv2.typing.MatLike, factor: int):
    ratio = 1.0 / factor
    return cv2.resize(
        image,  # original image
        (0, 0),  # set fx and fy, not the final size
        fx=ratio,
        fy=ratio,
        interpolation=cv2.INTER_AREA,
    )


erode_size = 3
erode_shape = cv2.MORPH_ELLIPSE
erode_element = cv2.getStructuringElement(
    erode_shape,
    (2 * erode_size + 1, 2 * erode_size + 1),
    (erode_size, erode_size),
)

grow_size = 10
grow_shape = cv2.MORPH_ELLIPSE
grow_element = cv2.getStructuringElement(
    grow_shape,
    (2 * grow_size + 1, 2 * grow_size + 1),
    (grow_size, grow_size),
)


def extract_objects(image: cv2.typing.MatLike, dpi: int) -> list[cv2.typing.MatLike]:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grayscale, 254, 255, cv2.THRESH_BINARY_INV)
    erode = cv2.erode(thresh, erode_element)
    dilate = cv2.dilate(erode, grow_element)
    blurred = cv2.GaussianBlur(dilate, (15, 15), 0)
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for i in range(len(contours)):
        rect = cv2.boundingRect(contours[i])
        x, y, width, height = rect
        if width / dpi > CARD_WIDTH and height / dpi > CARD_WIDTH:
            crop = image[y : (y + height), x : (x + width)]
            if width > height:
                crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
            objects.append(crop)
    return objects


def find_lines(
    image: cv2.typing.MatLike, downsample: int, kernel_size: int = 9
) -> list[Line] | None:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grayscale, 70, 150, cv2.THRESH_BINARY)
    thresh = cv2.GaussianBlur(thresh, (kernel_size, kernel_size), 0)
    _, thresh = cv2.threshold(thresh, 70, 150, cv2.THRESH_BINARY)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(thresh, low_threshold, high_threshold)

    # debug_show("Base Image", image, False)
    # debug_show("Threshold", thresh, False)
    debug_show("Edges", edges, True)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 360  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 500 // downsample  # minimum number of pixels making up a line
    max_line_gap = 60  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines_mat = cv2.HoughLinesP(
        edges, rho, theta, threshold, None, min_line_length, max_line_gap
    )

    if lines_mat is not None:
        lines = []
        for line in lines_mat:
            for x1, y1, x2, y2 in line:
                lines.append(Line(Vec(x1, y1), Vec(x2, y2)))
        return lines
    return None


def find_boundaries(
    image: cv2.typing.MatLike, lines: list[Line], dpi: int
) -> BoundsLines | None:
    target_width = CARD_WIDTH - 2 * BORDER_WIDTH if cli.center else CARD_WIDTH
    target_height = CARD_HEIGHT - 2 * BORDER_HEIGHT if cli.center else CARD_HEIGHT

    vertical_lines = {}
    horizontal_lines = {}
    for i, lhs in enumerate(lines[:-1]):
        for j in range(i + 1, len(lines) - 1):
            rhs = lines[j]
            if lhs.parallel(rhs):
                dist = lhs.dist(rhs) / dpi
                if lhs.vertical and abs(dist - target_width) < 0.01:
                    lhs_x = (lhs.from_pos.x + lhs.to_pos.x) / 2
                    rhs_x = (rhs.from_pos.x + rhs.to_pos.x) / 2
                    pair_x = min(lhs_x, rhs_x)

                    new_list = True
                    for x, vlines in vertical_lines.items():
                        if abs(x - pair_x) < 0.01:
                            vlines.add(i)
                            vlines.add(j)
                            new_list = False
                            break

                    if new_list:
                        vertical_lines[pair_x] = set([i, j])
                elif lhs.horizontal and abs(dist - target_height) < 0.01:
                    lhs_y = (lhs.from_pos.y + lhs.to_pos.y) / 2
                    rhs_y = (rhs.from_pos.y + rhs.to_pos.y) / 2
                    pair_y = min(lhs_y, rhs_y)

                    new_list = True
                    for y, hlines in horizontal_lines.items():
                        if abs(y - pair_y) < 0.01:
                            hlines.add(i)
                            hlines.add(j)
                            new_list = False
                            break

                    if new_list:
                        horizontal_lines[pair_y] = set([i, j])

    for k, v in list(vertical_lines.items()):
        vertical_lines[k] = [lines[i] for i in v]
    for k, v in list(horizontal_lines.items()):
        horizontal_lines[k] = [lines[i] for i in v]

    h, w = image.shape[:2]

    if len(vertical_lines) > 1:

        def get_center_deviation(lines: list[Line]):
            points = list(
                itertools.chain.from_iterable([(l.from_pos, l.to_pos) for l in lines])
            )
            x_coords = [int(p.x) for p in points]
            average_x = sum(x_coords) / len(x_coords)
            return abs(w / 2 - average_x)

        lists = list(vertical_lines.values())
        deviations = list(map(get_center_deviation, lists))
        vertical_lines = lists[np.argmin(deviations)]
    else:
        vertical_lines = list(vertical_lines.values())[0]

    if len(horizontal_lines) > 1:

        def get_center_deviation(lines: list[Line]):
            points = list(
                itertools.chain.from_iterable([(l.from_pos, l.to_pos) for l in lines])
            )
            y_coords = [int(p.y) for p in points]
            average_y = sum(y_coords) / len(y_coords)
            return abs(h / 2 - average_y)

        lists = list(horizontal_lines.values())
        deviations = list(map(get_center_deviation, lists))
        horizontal_lines = lists[np.argmin(deviations)]
    else:
        horizontal_lines = list(horizontal_lines.values())[0]

    if len(vertical_lines) < 2 or len(horizontal_lines) < 2:
        return None

    def average_lines(lines: list[Line]):
        slopes = [l.slope for l in lines]
        intercepts = [l.intercept for l in lines]

        average_slope = sum(slopes) / len(slopes)
        average_intercept = sum(intercepts) / len(intercepts)

        is_vertical = lines[0].vertical
        if is_vertical:
            x0 = average_intercept
            x1 = average_intercept + average_slope * h
            return Line(Vec(x0, 0), Vec(x1, h))
        else:
            y0 = average_intercept
            y1 = average_intercept + average_slope * w
            return Line(Vec(0, y0), Vec(w, y1))

    if len(vertical_lines) > 2:
        top_lines = [l for l in vertical_lines if l.intercept > w / 2]
        bottom_lines = [l for l in vertical_lines if l.intercept < w / 2]
        vertical_lines = [average_lines(top_lines), average_lines(bottom_lines)]
    elif vertical_lines[0].intercept > vertical_lines[1].intercept:
        vertical_lines = [vertical_lines[1], vertical_lines[0]]

    if len(horizontal_lines) > 2:
        left_lines = [l for l in horizontal_lines if l.intercept < h / 2]
        right_lines = [l for l in horizontal_lines if l.intercept > h / 2]
        horizontal_lines = [average_lines(left_lines), average_lines(right_lines)]
    elif horizontal_lines[0].intercept > horizontal_lines[1].intercept:
        horizontal_lines = [horizontal_lines[1], horizontal_lines[0]]

    def find_intersection(lhs: Line, rhs: Line) -> Vec:
        assert lhs.vertical != rhs.vertical
        if lhs.vertical:
            lhs, rhs = rhs.lhs

        # Solve for x and y
        # y = m * x + c
        # x = n * y + k
        # -> y = (n * c + k) / (1 - m * n)
        y = (lhs.slope * rhs.intercept + lhs.intercept) / (1 - lhs.slope * rhs.slope)
        x = rhs.slope * y + rhs.intercept
        return Vec(x, y)

    left_line, right_line = vertical_lines
    top_line, bottom_line = horizontal_lines

    top_left = find_intersection(top_line, left_line)
    top_right = find_intersection(top_line, right_line)
    bottom_right = find_intersection(bottom_line, right_line)
    bottom_left = find_intersection(bottom_line, left_line)

    left_line = Line(top_left, bottom_left)
    bottom_line = Line(bottom_left, bottom_right)
    right_line = Line(bottom_right, top_right)
    top_line = Line(top_right, top_left)

    return (left_line, right_line), (top_line, bottom_line)


def debug_show(title: str, image: cv2.typing.MatLike, wait_key: bool = True):
    global DISPLAY_DOWNSAMPLE
    if DISPLAY_DOWNSAMPLE > 1:
        image = downsample_image(image, DISPLAY_DOWNSAMPLE)

    cv2.imshow(title, image)
    if wait_key:
        cv2.waitKey()


def debug_show_lines(image: cv2.typing.MatLike, lines: list[Line] | BoundsLines):
    global DISPLAY_DOWNSAMPLE

    if isinstance(lines, tuple):
        lines = list(lines[0]) + list(lines[1])

    line_image = np.copy(image)
    for line in lines:
        cv2.line(
            line_image,
            (int(line.from_pos.x), int(line.from_pos.y)),
            (int(line.to_pos.x), int(line.to_pos.y)),
            (0, 0, 255),
            int(DISPLAY_DOWNSAMPLE),
        )

    debug_show("Found Lines", line_image, True)


def extract_transform(
    image: cv2.typing.MatLike, bounds: BoundsLines, dpi: int
) -> Transform:
    angle = -math.atan(bounds[0][0].slope)

    h, w = image.shape[:2]
    pivot = Vec(w // 2, h // 2)
    rotated_bounds = (
        (
            bounds[0][0].rotate(angle, pivot),
            bounds[0][1].rotate(angle, pivot),
        ),
        (
            bounds[1][0].rotate(angle, pivot),
            bounds[1][1].rotate(angle, pivot),
        ),
    )

    lines = list(rotated_bounds[0]) + list(rotated_bounds[1])
    points = list(
        itertools.chain.from_iterable([(l.from_pos, l.to_pos) for l in lines])
    )
    x_coords = [int(p.x) for p in points]
    y_coords = [int(p.y) for p in points]

    offset = Vec(min(x_coords), min(y_coords))
    if cli.center:
        offset.x -= BORDER_WIDTH * dpi
        offset.y -= BORDER_HEIGHT * dpi

    return Transform(angle, offset)


def apply_transform(
    image: cv2.typing.MatLike, transform: Transform, dpi: int
) -> cv2.typing.MatLike:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, transform.rotation * 180 / np.pi, 1.0)
    rotated_image = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    offset = Vec(int(transform.translation.x), int(transform.translation.y))
    size = Vec(int(CARD_WIDTH * dpi), int(CARD_HEIGHT * dpi))

    cropped_image = rotated_image[
        offset.y : offset.y + size.y, offset.x : offset.x + size.x
    ]
    return cropped_image


def write_image(image: cv2.typing.MatLike, path: Path):
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(path, image)


# TODO
def apply_border(img, border_size, border_color):
    if border_size <= 0 or border_color is None:
        return img
    result = img.copy()
    h, w = result.shape[:2]
    border = min(border_size, h // 4, w // 4)
    result[0:border, :] = border_color
    result[h - border : h, :] = border_color
    result[:, 0:border] = border_color
    result[:, w - border : w] = border_color
    return result


def main():
    print(f"{cli.program}{' - Verbose' if cli.verbose else ''}")
    print_verbose(f"Running with input {cli.input} and output {cli.output}...")

    output = Path(cli.output)
    if output.exists() and not output.is_dir():
        print(f"Error: Output '{cli.output}' not a folder")
        return 1
    elif not output.exists():
        os.makedirs(output)

    input = Path(cli.input)
    if pdfs := get_pdfs(input):
        print_verbose(f"Found {len(pdfs)} pdf files, proceeding to extract images...")

        interim_path = output / "interim"
        if not interim_path.exists():
            os.makedirs(interim_path)

        extract = functools.partial(extract_pdf_images, interim=interim_path)
        pdf_images = list(itertools.chain.from_iterable(map(extract, pdfs)))
    else:
        pdf_images = []

    images = get_images(input) + pdf_images
    if not images:
        print(f"Error: No images in {input}")
        return 1

    dpi = 1200 / cli.downsample

    if cli.display_downsample > cli.downsample:
        global DISPLAY_DOWNSAMPLE
        DISPLAY_DOWNSAMPLE = cli.display_downsample / cli.downsample

    for image_path in images:
        print_verbose(f"Attempting to extract cards from image {image_path.name}...")

        image = cv2.imread(image_path)
        if cli.downsample > 1:
            print_verbose(f"\tDownsampling...")
            image = downsample_image(image, cli.downsample)

        print_verbose("\tRoughly extracting objects...")
        objects = extract_objects(image, dpi)
        print(f"\tFound {len(objects)} objects...")

        for i, obj in enumerate(objects):
            print(f"\tHandling object {i}...")

            print_verbose(f"\t\tFinding lines...")
            if lines := find_lines(obj, cli.downsample):
                debug_show_lines(obj, lines)
                if bounds := find_boundaries(obj, lines, dpi):
                    debug_show_lines(obj, bounds)

                    transform = extract_transform(obj, bounds, dpi)
                    transformed = apply_transform(obj, transform, dpi)

                    if cli.center:
                        h, w = transformed.shape[:2]
                        border_left = BORDER_WIDTH * dpi
                        border_right = CARD_WIDTH * dpi - border_left
                        border_top = BORDER_HEIGHT * dpi
                        border_bottom = CARD_HEIGHT * dpi - border_top
                        ideal_lines = [
                            Line(Vec(border_left, 0), Vec(border_left, h)),
                            Line(Vec(border_right, 0), Vec(border_right, h)),
                            Line(Vec(0, border_top), Vec(w, border_top)),
                            Line(Vec(0, border_bottom), Vec(w, border_bottom)),
                        ]
                        debug_show_lines(transformed, ideal_lines)
                    else:
                        debug_show("Found Lines", transformed)

                    write_image(transformed, output / f"{image_path.stem}_{i}.png")
                else:
                    print("\t\tCould not determine card boundaries...")
            else:
                print("\t\tCould not find any lines, likely not a card...")

    return 0


if __name__ == "__main__":
    exit(main())
