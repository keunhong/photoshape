#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo $DIR

pyenv activate terial

export PYTHONPATH=$DIR/src:$PYTHONPATH
export PYTHONPATH=$DIR/thirdparty/pyhog:$PYTHONPATH
export PYTHONPATH=$DIR/thirdparty/rendkit:$PYTHONPATH
export PYTHONPATH=$DIR/thirdparty/kitnn:$PYTHONPATH
export PYTHONPATH=$DIR/thirdparty/toolbox:$PYTHONPATH
export PYTHONPATH=$DIR/thirdparty/vispy:$PYTHONPATH
export PYTHONPATH=$DIR/thirdparty/brender:$PYTHONPATH

