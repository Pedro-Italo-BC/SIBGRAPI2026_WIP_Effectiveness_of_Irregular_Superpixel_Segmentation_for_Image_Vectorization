
import cv2
import numpy as np
import argparse


def segment_seeds(image_path, num_superpixels=2000, num_levels=4, iterations=10, output_path="labels.png"):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Was not possible to open image: {image_path}")

    height, width = image.shape[:2]

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    seeds = cv2.ximgproc.createSuperpixelSEEDS(
        width,
        height,
        3,
        num_superpixels,
        num_levels,
        2,
        5,
        False
    )

    seeds.iterate(lab, iterations)

    labels = seeds.getLabels()

    cv2.imwrite(output_path, labels.astype(np.uint16))

    contour_mask = seeds.getLabelContourMask(thick_line=True)
    visualization = image.copy()
    visualization[contour_mask != 0] = (0, 0, 255)

    # cv2.imwrite("seeds_visualization.png", visualization)
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--out", default="labels.png")

    args = parser.parse_args()

    segment_seeds(
        args.img,
        args.n,
        args.levels,
        args.iterations,
        args.out
    )