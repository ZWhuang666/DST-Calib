from torch.utils.cpp_extension import load
import os
import sys
print(sys.path)
# Resolve the current script directory.
script_dir = os.path.dirname(__file__)
build_dir = 'build'
os.makedirs(build_dir, exist_ok=True)
# Specify the extension source files.
sources = [
    os.path.join(script_dir, "chamfer_distance.cpp"),
    os.path.join(script_dir, "chamfer_distance.cu"),
]

# Compile and load the C++/CUDA extension.
#cd = load(name="cd", sources=sources)
cd = load(name="cd", sources=sources, build_directory='build')
# Print the loaded module for verification.
print(cd)
