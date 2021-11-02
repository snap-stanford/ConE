cd ..

bash run.sh train RotC DDB14 0 1 1024 50 500 0.01 0.5 0.002 40000 4 -de \
    --tail_batch_only --do_valid --valid_steps 20000 --save_checkpoint 40000 \
    --train_with_relation_category --lr_decay_epoch "20000" --do_test_relation_category

bash run.sh train ConE DDB14 0 1 1024 50 500 10 0.5 0.001 40000 2 -de \
    --tail_batch_only --do_valid --valid_steps 20000 --save_checkpoint 40000 \
    --train_with_relation_category --uni_weight --lr_decay_epoch "30000" \
    --do_test_relation_category --cone_penalty --fix_att 50 \
    --w 0.7 --pretrained "./models/RotC_DDB14_1/checkpoint/ckpt_39999"