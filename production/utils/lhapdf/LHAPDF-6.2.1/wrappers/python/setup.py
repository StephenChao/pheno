#! /usr/bin/env python

import os
from distutils.core import setup
from glob import glob
from distutils.extension import Extension

incdir_src = os.path.abspath("../../include")
incdir_build = os.path.abspath("../../include")
libdir = os.path.abspath("../../src/.libs")


## Configure the C++ extension and LHAPDF package
ext = Extension("lhapdf",
                ["lhapdf.cpp"],
                include_dirs = [incdir_src, incdir_build],
                extra_compile_args=["-I/scratchfs/cms/licq/utils/lhapdf/lhapdf-install-py2/include"],
                library_dirs = [libdir],
                language = "C++",
                libraries = ["stdc++", "LHAPDF"])
setup(name = "LHAPDF",
      version = "6.2.1",
      ext_modules = [ext])


# ## Also install a lightweight YAML parser
# setup(
#     name='poyo',
#     version='0.4.1',
#     author='Raphael Pierzina',
#     author_email='raphael@hackebrot.de',
#     license='MIT',
#     url='https://github.com/hackebrot/poyo',
#     description='A lightweight YAML Parser for Python',
#     packages=['poyo'],
#     package_dir={'poyo': 'poyo'},
#     keywords=['YAML', 'parser'],
# )
