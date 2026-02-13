cd docs
rm -rf build

# update api rst
#rm -rf source/api/
#sphinx-apidoc --module-first -o source/api/ ../modelscope/
make html
