import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plot_utils as putl
from config.paths import MODELS_DIR

print(MODELS_DIR)
file_path = os.path.join(MODELS_DIR, "with_noise")
file_handler = putl.ModelFileHandler(file_path)

tags = file_handler.get_logged_tags(writeout=True)

scalars = {}

for t in tags:
    steps, vals = file_handler.get_tag_scalars(t)
    scalars[t] = vals

putl.plot(steps, scalars[tags[0]])