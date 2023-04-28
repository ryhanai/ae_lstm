# -*- coding: utf-8 -*-

import os
import glob
import subprocess


def gen_collada_with_rotation(base_directory, rotation_script = 'rotate.mlx'):
    obj_files = glob.glob(os.path.join(base_directory, 'VER002/Seamless/vt2048/*/*.obj'))
    for obj_file in obj_files:
        name = os.path.splitext(os.path.split(obj_file)[-1])[0]
        dae_file = os.path.join('collada_files', name + '.obj')
        cmd = ['meshlabserver', '-i', obj_file, '-o', dae_file, '-s', rotation_script]
        print(' '.join(cmd))
        subprocess.call(cmd)
