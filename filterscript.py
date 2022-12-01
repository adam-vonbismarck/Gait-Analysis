import os, re, os.path
import shutil

path = ""

for root, dirs, files in os.walk(path):
    for dir in dirs:
        if len(dir) == 3 and dir != "090" and len(root) > len(path) + 5:
            shutil.rmtree(os.path.join(root, dir))
