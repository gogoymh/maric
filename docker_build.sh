docker build -t maric:latest .
docker run -itd --rm --gpus all --ipc=host -v $(pwd):/app --name maric maric:latest bash