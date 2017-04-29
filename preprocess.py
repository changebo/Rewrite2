# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import scipy.misc as misc
import os
from PIL import Image
#import Image
from PIL import ImageDraw
#import ImageDraw
from PIL import ImageFont
#import ImageFont

reload(sys)
sys.setdefaultencoding("utf-8")

FLAGS = None

def render_fonts_image(x, path, img_per_row, unit_scale=True):
    if unit_scale:
        # scale 0-1 matrix back to gray scale bitmaps
        bitmaps = (x * 255.).astype(dtype=np.int16) % 256
    else:
        bitmaps = x
    num_imgs, w, h = x.shape
    assert w == h
    side = int(w)
    width = img_per_row * side
    height = int(np.ceil(float(num_imgs) / img_per_row)) * side
    canvas = np.zeros(shape=(height, width), dtype=np.int16)
    # make the canvas all white
    canvas.fill(255)
    for idx, bm in enumerate(bitmaps):
        x = side * int(idx / img_per_row)
        y = side * int(idx % img_per_row)
        canvas[x: x + side, y: y + side] = bm
    misc.toimage(canvas).save(path)
    return path

def draw_char_bitmap(ch, font, char_size, x_offset, y_offset):
    image = Image.new("RGB", (char_size, char_size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    gray = image.convert('L')
    bitmap = np.asarray(gray)
    return bitmap


def generate_font_bitmaps(chars, font_path, char_size, canvas_size, x_offset, y_offset):
    font_obj = ImageFont.truetype(font_path, char_size)
    bitmaps = list()
    for c in chars:
        bm = draw_char_bitmap(c, font_obj, canvas_size, x_offset, y_offset)
        bitmaps.append(bm)
    return np.array(bitmaps)


def process_font(chars, font_path, save_dir, x_offset=0, y_offset=0, mode='target', char_size=64, canvas=70):
    # char_size = 64
    # canvas = 80

    # if mode == 'source':
    #     char_size = char_size * 2
    #     canvas = canvas * 2
    font_bitmaps = generate_font_bitmaps(chars, font_path, char_size,
                                         canvas, x_offset, y_offset)
    _, ext = os.path.splitext(font_path)
    if not ext.lower() in [".otf", ".ttf"]:
        raise RuntimeError("unknown font type found %s. only TrueType or OpenType is supported" % ext)
    _, tail = os.path.split(font_path)
    font_name = ".".join(tail.split(".")[:-1])
    bitmap_path = os.path.join(save_dir, "%s.npy" % font_name)
    np.save(bitmap_path, font_bitmaps)
    sample_image_path = os.path.join(save_dir, "%s_sample.png" % font_name)
    render_fonts_image(font_bitmaps[:100], sample_image_path, 10, False)
    print("%s font %s saved at %s" % (mode, font_name, bitmap_path))


def get_chars_set(path):
    """
    Expect a text file that each line is a char
    """
    chars = list()
    with open(path) as f:
        for line in f:
            line = u"%s" % line
            char = line.split()[0]
            chars.append(char)
    return chars


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_font', type=str, default=None,
                        help='npy bitmap for the source font')
    parser.add_argument('--target_font', type=str, default=None,
                        help='npy bitmap for the target font')
    parser.add_argument('--char_list', type=str, required=True,
                        help='source file for chars. each line contains one char')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='directory to save output')
    parser.add_argument('--sx', type=int, default=0,
                        help='source font x offset')
    parser.add_argument('--sy', type=int, default=0,
                        help='source font y offset')
    parser.add_argument('--tx', type=int, default=0,
                        help='target font x offset')
    parser.add_argument('--ty', type=int, default=0,
                        help='target font y offset')
    parser.add_argument('--char_size', type=int, default=32,
                        help='char_size. Default = 32')
    parser.add_argument('--canvas', type=int, default=32,
                        help='canvas. Default = 32')
    FLAGS = parser.parse_args()

    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    chars = get_chars_set(FLAGS.char_list)
    if FLAGS.source_font:
        process_font(chars, FLAGS.source_font, FLAGS.save_dir, FLAGS.sx, FLAGS.sy, 
            mode='source', char_size=FLAGS.char_size, canvas=FLAGS.canvas)
    if FLAGS.target_font:
        process_font(chars, FLAGS.target_font, FLAGS.save_dir, FLAGS.tx, FLAGS.ty, 
            mode='target', char_size=FLAGS.char_size, canvas=FLAGS.canvas)