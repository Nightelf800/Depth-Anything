import sys


_LIBS = ['./VFDepth/external/packnet_sfm', './VFDepth/external/dgp']

def setup_env():
    if not _LIBS[0] in sys.path:
        for lib in _LIBS:
            sys.path.append(lib)

setup_env()