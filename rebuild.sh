#!/bin/bash
pip uninstall mega-core
python setup.py clean
rm -r build/
rm -r mega_core.egg-info/
rm mega_core/_C.cpython-3*
python setup.py build develop
