#!/bin/sh

#update version
version=$(cat pyproject.toml | grep "version" | cut -d '"' -f 2)
major=$(echo $version | cut -d '.' -f 1)
minor=$(echo $version | cut -d '.' -f 2)
patch=$(echo $version | cut -d '.' -f 3)
new_patch=$((patch+1))
new_version="version = \"$major.$minor.$new_patch\""

(cat pyproject.toml | head -n 2 ; echo $new_version ; cat pyproject.toml | tail -n +4) > pyproject.toml.new
mv pyproject.toml.new pyproject.toml

#update binaries
rm -rf dist
python -m build
python -m twine upload dist/*

#update git
git add -A
git commit -m "update to version $new_version"
git push

#update local
sleep 5
pip3 install --upgrade micrograd-utils
