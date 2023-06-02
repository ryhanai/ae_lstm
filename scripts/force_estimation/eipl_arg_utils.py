#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
# 

import json
import datetime
import os
# from .print_func import *
# from .path_utils  import *


OK = '\033[92m'
WARN = '\033[93m'
NG = '\033[91m'
END_CODE = '\033[0m'

def print_info(msg):
    print( OK + "[INFO] " + END_CODE + msg )

def print_warn(msg):
    print( WARN + "[WARNING] " + END_CODE +  msg )

def print_error(msg):
    print( NG + "[ERROR] " + END_CODE + msg )



def check_filename(filename):
    if os.path.exists(filename):
        raise ValueError("{} exists.".format(filename))
    return filename

def check_path(path, mkdir=False):
    """
    Checks that path is collect
    """
    if path[-1] == '/':
        path = path[:-1]

    if not os.path.exists(path):
        if mkdir:
            os.makedirs(path, exist_ok=True)             
        else:
            raise ValueError('%s does not exist' % path)
    
    return path



def print_args(args):
    """ Print arguments """
    if not isinstance(args, dict):
        args = vars(args)

    keys = args.keys() 
    keys = sorted(keys)

    print("================================")
    for key in keys:
        print("{} : {}".format(key, args[key]))
    print("================================")

    
def save_args(args, filename):
    """ Dump arguments as json file """
    with open(filename, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

        
def restore_args(filename):
    """ Load argument file from file """
    with open(filename, 'r') as f:
        args = json.load(f)
    return args


def get_config(args, tag, default=None):
    """ Get value from argument """
    if not isinstance(args, dict):
        raise ValueError("args should be dict.")
    
    if tag in args:
        if args[tag] is None:
            print_info("set {} <-- {} (default)".format(tag, default))
            return default
        else:
            print_info("set {} <--- {}".format(tag, args[tag]))
            return args[tag]
    else:
        if default is None:
            raise ValueError("you need to specify config {}".format(tag))
        
        print_info("set {} <-- {} (default)".format(tag, default))
        return default

    
def check_args(args):
    """ Check arguments """

    if args.tag is None:
        tag = datetime.datetime.today().strftime("%Y%m%d_%H%M_%S")
        args.tag = tag
        print_info('Set tag = %s' % tag)
        
    # make log directory
    check_path(os.path.join(args.log_dir, args.tag), mkdir=True)

    # saves arguments into json file
    save_args(args, os.path.join(args.log_dir, args.tag, 'args.json'))
    
    print_args(args)
    return args
