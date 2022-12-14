import os
import shutil
import config

for root, dirs, files in os.walk(config.Parameters.casia_b_path):
    for dir in dirs:
        if len(dir) == 3 and dir != "090" and len(root) > len(config.Parameters.casia_b_path) + 5:
            shutil.rmtree(os.path.join(root, dir))
