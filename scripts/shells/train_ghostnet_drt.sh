python ../distribute_train_script.py --data-path /s4/piston/ImageNet --batch-size 256 --dtype float16 \
                                     -j 48 --epochs 360 --lr 2.6 --warmup-epochs 5 --label-smoothing \
                                     --no-wd --wd 0.00003 --model GhostNet --log-interval 150 --model-info \
                                     --dist-url tcp://127.0.0.1:26548 --world-size 1 --rank 0