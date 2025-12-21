docker build -f docker/Dockerfile -t pychop-jupyter .
docker run --rm -p 8888:8888 pychop-jupyter
