rm -rf dist
poetry build --format wheel
pip3 uninstall torch_flatbuffers -y
pip3 install dist/* --upgrade