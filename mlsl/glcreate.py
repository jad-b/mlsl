#!/usr/bin/env python2
"""
graphlab
========
Assists with registering, loading, and configuring GraphLab by Dato.

Note the python 2 shebang at the top of this file. At the time of this writing,
Graphlab does not support Python 3. - jdb, 2016May09
"""
import argparse
import os
import sys
import time


class VersionError(Exception):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def load_graphlab():
    if sys.version_info >= (3, 0):
        raise VersionError("Graphlab is only available in Python 2")
    start = time.clock()  # noqa
    import graphlab
    gl_product_key = os.getenv('GLCREATE_PRODUCT_KEY', False)
    if not gl_product_key:
        print("Please set GLCREATE_PRODUCT_KEY")
        return

    graphlab.product_key.set_product_key(gl_product_key)
    # Display graphlab canvas in notebook
    graphlab.canvas.set_target('ipynb')
    # Number of workers
    graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 16)
    since = time.clock() - start
    print("Graphlab loaded in {:.3f} seconds".format(since))
    return graphlab


def convert_to_csv(filename):
    gl = load_graphlab()
    sframe = gl.SFrame(filename)
    noext_filename, _ = os.path.splitext(os.path.abspath(filename))
    new_filename = noext_filename + '.csv'
    df = sframe.to_dataframe()
    df.to_csv(new_filename)
    assert os.path.exists(new_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file',
                        help='GraphLab file to convert to Pandas .csv')
    args = parser.parse_args()
    convert_to_csv(args.file)
