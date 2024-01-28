docker build -t listen_docker_image .
docker stop listen_container
docker rm listen_container
docker run -it --name listen_container -p 8000:8000 listen_docker_image
