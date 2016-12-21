import numpy as np
from pylab import *
import tempfile
import matplotlib
import os
from os.path import join, exists, basename, splitext
# import cv2
import csv
import glob
import subprocess
import glob
import shutil

from wikipaintings_finetuning import process_scores
from plot_scores import plot_scores


input_dir = '/home/ariel/Documents/tesis/tests/inputs/'
content_image_dir = input_dir + 'content_images/'
style_image_dir = input_dir + 'style_images/'

output_dir = '/home/ariel/Documents/tesis/tests/outputs/'

def generate_output(content_image, style_image, output, start):
    # subprocess.check_output(['cd', 'neural-style/'])
    subprocess.check_output(['th', 'neural_style.lua', '-content_image', content_image, 
        '-style_image', style_image, '-output_image', output, '-init', start])
    # subprocess.check_output(['cd', '..'])

# generate_output(content_image='neural-style/examples/inputs/golden_gate.jpg',
#         style_image='neural-style/examples/inputs/woman-with-hat-matisse.jpg',
#         output='gate_matisse/gate_matisse.png',
#         start='random')

def run_full_pipeline():
    for content_image in glob.iglob(content_image_dir+'*'):
        content_image_name = splitext(basename(content_image))[0] 
        for style, subdir, style_images in os.walk(style_image_dir):
            style_name = basename(style)
            for style_image in style_images:
                style_image_name = splitext(style_image)[0]
                output_name = content_image_name + '_' + style_image_name
                try:
                    os.mkdir(join(output_dir, style_name, output_name))
                except:
                    pass
                # os.rmdir(join(output_dir, style_name, output_name))
                for start in ['image', 'random']:
                    start_dir = join(output_dir, style_name, output_name, start)
                    try:
                        os.mkdir(start_dir)
                    except:
                        pass
                    # os.rmdir(start_dir)
                    output = start_dir + '/'+output_name+ '.png'
                    style_image_in = style +'/'+ style_image
                    generate_output(content_image, style_image_in, output, start)
                    print('STYLE NAME', style_name)
                    print('output name', output_name)
                    base_image = splitext(output)[0] + '_10.png'
                    init_image = splitext(output)[0] + '_0.png' 
                    shutil.copy(base_image, init_image)
                    process_scores(output_path=start_dir, output_name=output_name)
                    plot_scores(path=start_dir, image_name=output_name)


run_full_pipeline()