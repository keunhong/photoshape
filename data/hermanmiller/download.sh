#!/bin/bash

echo "Downloading models from Herman Miller..."

if [ ! -d "zip" ]; then
  mkdir zip
fi
cd zip && xargs -n 1 curl -L -O < ../urls.txt && cd ..

echo "Extracted downloaded models..."

if [ ! -d "extracted" ]; then
  mkdir extracted
fi
unzip -C -d extracted "zip/*.zip" \*.3ds

echo "Done."
