cd ..

bash run.sh train RotC FB15K237 0 1 1024 100 500 3 0.5 0.0001 160000 4 -de \
    --do_valid --valid_steps 20000 --save_checkpoint 20000 --sum_loss \
    --train_with_relation_category --lr_decay_epoch "100000" --do_test_relation_category