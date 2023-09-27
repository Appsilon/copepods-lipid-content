from pathlib import Path
from typing import List, Tuple, Callable

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from PIL import Image, ImageDraw


def generic_image_path(root: Path):
    def image_path(img: str):
        return root / "data" / "images" / img

    return image_path


def generic_image_path_processed(root: Path):
    def image_path_processed(img: str):
        return root / "data" / "images_processed" / img

    return image_path_processed


def generic_segmentation_path(root: Path):
    def segmentation_path(img: str):
        return root / "data" / "segmentations" / img

    return segmentation_path


def generic_segmentation_path_processed(root: Path):
    def segmentation_path_processed(img: str):
        return root / "data" / "segmentations_processed" / img

    return segmentation_path_processed


def add_mask(source: Image.Image, mask: Image.Image) -> Image.Image:
    source = source.convert("RGBA")
    mask = Image.fromarray(np.r_[mask] * 255).convert("RGBA")
    M = np.r_[mask]
    M[:, :, 1:2] = 0
    M[:, :, 3] = 120

    mask = Image.fromarray(M)
    return Image.alpha_composite(source, mask)


def preview_raw(
    df: pd.DataFrame, index: int, image_path_function, segmentation_path_function
) -> Image.Image:
    row = df.iloc[index, :]
    img = Image.open(image_path_function(row.filename))
    img_seg = Image.open(segmentation_path_function(row.filename))
    return add_mask(img, img_seg)


def read_copypods_df(root, segmentation_csv_path, all_image_data_csv_path):
    df_seg = pd.read_csv(root / segmentation_csv_path)
    df_seg["id"] = df_seg["filename"].str.slice(stop=-4)
    df_seg["weight"] = df_seg["shape"].apply(lambda s: eval(s)[0])
    df_seg["height"] = df_seg["shape"].apply(lambda s: eval(s)[1])
    df_seg["segmentation"] = df_seg["segmentation"].apply(
        lambda s: [(x, y) for (x, y) in eval(s)[0]]
    )
    df_seg.drop(columns=["Unnamed: 0", "number_copepods", "shape"], inplace=True)
    df_seg = df_seg.query(
        'filename != "20130815 104419 867 000000 0996 0096(1).bmp"'
    ).reset_index(drop=True)

    df_all = pd.read_csv(root / all_image_data_csv_path)
    df_seg = df_seg.merge(df_all, left_on="id", right_on="new_image")
    return df_seg


def draw_segmentation(img: Image.Image, verticies: List[Tuple]) -> Image.Image:
    ims = Image.new("L", img.size)
    draw = ImageDraw.Draw(ims)
    draw.polygon(verticies, fill=1)
    return ims


def paste_imgs(image1, image2):
    image1_size = image1.size
    image2_size = image2.size
    new_image = Image.new(
        "RGB",
        (image1_size[0] + image2_size[0], max(image1_size[1], image2_size[1])),
        "teal",
    )
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1_size[0], 0))
    return new_image


def pad_image_topleft(img: Image.Image, ph: int = 600, pw: int = 600) -> Image.Image:
    img_processed = Image.new("L", (ph, pw))
    img_processed.paste(img, (0, 0))
    return img_processed


def pad_image_center(img: Image.Image, ph: int = 600, pw: int = 600) -> Image.Image:
    img_processed = Image.new("L", (ph, pw))
    h, w = img.size
    img_processed.paste(img, (int(ph / 2 - h / 2), int(pw / 2 - w / 2)))
    return img_processed


def enhance_image(
    img: Image.Image, enhancers: List[Callable[[ArrayLike], ArrayLike]]
) -> Image.Image:
    A = np.r_[img] / 255
    for enhancer in enhancers:
        A = enhancer(A)
    return Image.fromarray((A * 255).astype(np.uint8))
