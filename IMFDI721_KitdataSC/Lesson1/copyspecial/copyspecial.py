#!/usr/bin/python
# Copyright 2010 Google Inc.
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

# Google's Python Class
# http://code.google.com/edu/languages/google-python-class/

import sys
import re
import os
import shutil
import subprocess
import glob
import zipfile

"""Copy Special exercise
"""

# +++your code here+++
# Write functions and modify main() to call them
def get_special_paths(direction):
    files_in_path = os.listdir(direction)
    spec_list = []
    for f in files_in_path:
        if re.search(r'_{2}\w+_{2}',f):
            spec_list.append(direction+"\\"+f)
    return spec_list
    
def copy_to(paths,direction):
    #if direction doesn't exist, we create it
    if not os.path.exists(direction):
        os.mkdir(direction)
    for p in paths:
        file_c=get_special_paths(p)
        for f in file_c:
            shutil.copy(f,direction)

        
def zip_to(paths,zippath):
    temp=zippath+"\\temp"
    if not os.path.exists(zippath):
        os.mkdir(zippath)
    if not os.path.exists(temp):
        os.mkdir(temp)
    copy_to(paths,temp) #Copy of the files in a temp folder
    shutil.make_archive(zippath+"\\my_spec_zip","zip",temp)
    shutil.rmtree(temp)
        

def main():
  # This basic command line argument parsing code is provided.
  # Add code to call your functions below.

  # Make a list of command line arguments, omitting the [0] element
  # which is the script itself.
  args = sys.argv[1:]
  if not args:
    print("usage: [--todir dir][--tozip zipfile] dir [dir ...]")
    sys.exit(1)

  # todir and tozip are either set from command line
  # or left as the empty string.
  # The args array is left just containing the dirs.
  todir = ''
  if args[0] == '--todir':
    todir = args[1]
    del args[0:2]

  tozip = ''
  if args[0] == '--tozip':
    tozip = args[1]
    del args[0:2]

  if len(args) == 0:
    print("error: must specify one or more dirs")
    sys.exit(1)

  # +++your code here+++
  # Call your functions
  
if __name__ == "__main__":
  main()
