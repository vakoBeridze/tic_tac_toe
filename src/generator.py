import os
import sys
from random import randint

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GENERATED_IMAGES_DIR = "gen_images"
FONTS_DIR = "fonts/"


def create_dir_if_missing(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def parse_fonts_directory(fonts_path):
    font_files = os.listdir(fonts_path)

    fonts = []
    for f in font_files:
        parsed_font = ImageFont.truetype(os.path.join(fonts_path, f))
        fonts.append(parsed_font.font.family)

    return fonts


def generate(size):
    dir_x = create_dir_if_missing(os.path.join(GENERATED_IMAGES_DIR, 'X'))
    dir_0 = create_dir_if_missing(os.path.join(GENERATED_IMAGES_DIR, '0'))

    font_files = os.listdir(FONTS_DIR)

    for i in range(size):
        img = Image.new('RGB', (30, 30), color=(255, 255, 255))
        font_size = randint(10, 18)
        random_font_file = np.random.choice(font_files)
        random_font = ImageFont.truetype(os.path.join(FONTS_DIR, random_font_file), font_size)
        d = ImageDraw.Draw(img)
        d.text((randint(5, 12), randint(5, 12)), "X", font=random_font, fill=(0, 0, 0))
        img.save(os.path.join(dir_x, 'X_{}.png'.format(i)))

    for i in range(size):
        img = Image.new('RGB', (30, 30), color=(255, 255, 255))
        font_size = randint(10, 18)
        random_font_file = np.random.choice(font_files)
        random_font = ImageFont.truetype(os.path.join(FONTS_DIR, random_font_file), font_size)
        d = ImageDraw.Draw(img)
        d.text((randint(5, 12), randint(5, 12)), "0", font=random_font, fill=(0, 0, 0))
        img.save(os.path.join(dir_0, '0_{}.png'.format(i)))


if __name__ == '__main__':
    if len(sys.argv) == 3:
        GENERATED_IMAGES_DIR = sys.argv[1]
        generate(int(sys.argv[2]))
    else:
        raise ValueError
