docker pull pkuzhou/gnnear_gpu:latest
nvidia-docker run --gpus 1  -it  --rm   -v $PWD:/workspace -v /etc/passwd:/etc/passwd --name=gnnear_eval pkuzhou/gnnear_gpu:latest