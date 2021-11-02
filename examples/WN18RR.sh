cd ..

bash run.sh train RotC wn18rr 0 1 1024 50 500 0.01 0.5 0.002 40000 8 -de \
    --tail_batch_only --do_valid --valid_steps 20000 --save_checkpoint 40000 \
    --train_with_relation_category --lr_decay_epoch "25000" --do_test_relation_category

bash run.sh train ConE wn18rr 0 1 1024 50 500 10 0.5 0.001 40000 4 -de \
    --tail_batch_only --do_valid --valid_steps 20000 --save_checkpoint 40000 \
    --train_with_relation_category --uni_weight --lr_decay_epoch "30000" \
    --do_test_relation_category --cone_penalty --fix_att 100 --do_classification \
    --do_lca 1 --w 0.5 --pretrained "./models/RotC_wn18rr_1/checkpoint/ckpt_39999"