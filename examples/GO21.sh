cd ..

bash run.sh train RotC GO21 0 1 1024 50 500 10 0.5 0.0005 80000 4 -de \
    --tail_batch_only --do_valid --valid_steps 20000 --save_checkpoint 80000 --sum_loss \
    --train_with_relation_category --lr_decay_epoch "30000,60000" --do_test_relation_category

bash run.sh train ConE GO21 0 1 1024 50 500 10 0.5 0.005 80000 2 -de \
    --tail_batch_only --do_valid --valid_steps 20000 --save_checkpoint 80000 \
    --train_with_relation_category --uni_weight --lr_decay_epoch "30000,60000" \
    --do_test_relation_category --cone_penalty --fix_att 50 \
    --w 0.5 --pretrained "./models/RotC_GO21_1/checkpoint/ckpt_79999"