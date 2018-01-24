#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export ROOT_DIR="${ROOT_DIR}"

MODULES='baselines'

for module in $MODULES
do
    module_path=$ROOT_DIR/$module
    echo "adding module: $module_path"
    export PYTHONPATH=$module_path:$PYTHONPATH
done

cd $ROOT_DIR

echo "PYTHON PATH IS: $PYTHONPATH"