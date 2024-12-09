export CUDA_VISIBLE_DEVICES=$1
accelerate launch train.py ${@:2}
