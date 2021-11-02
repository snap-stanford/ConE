#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=codes
DATA_PATH=data
SAVE_PATH=models
TB_PATH=tensorboard

#The first four parameters must be provided
MODE=$1
MODEL=$2
DATASET=$3
GPU_DEVICE=$4
SAVE_ID=$5

FULL_DATA_PATH=$DATA_PATH/$DATASET
SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"_"$SAVE_ID"
TB_PATH=$TB_PATH/"$MODEL"_"$DATASET"_"$SAVE_ID"

#Only used in training
BATCH_SIZE=$6
NEGATIVE_SAMPLE_SIZE=$7
HIDDEN_DIM=$8
GAMMA=$9
ALPHA=${10}
LEARNING_RATE=${11}
MAX_STEPS=${12}
TEST_BATCH_SIZE=${13}


#Write command history for reproducibity
mkdir $SAVE
COMMAND_HISTORY_PATH=$SAVE/command_history
for var in "$@"
do
    echo "$var"
done > $COMMAND_HISTORY_PATH

#Copy scripts for reproducibity
cp ./$CODE_PATH/model.py ./$SAVE/
cp ./$CODE_PATH/run.py ./$SAVE/

if [ $MODE == "train" ]
then

echo "Start Training......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_train \
    --cuda \
    --data_path $FULL_DATA_PATH \
    --model $MODEL \
    -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
    -g $GAMMA -a $ALPHA -adv \
    -lr $LEARNING_RATE --max_steps $MAX_STEPS \
    -save $SAVE \
    --tb_path $TB_PATH \
    --test_batch_size $TEST_BATCH_SIZE \
    ${14} ${15} ${16} ${17} ${18} ${19} ${20} ${21} ${22} ${23} ${24} ${25} ${26} ${27} ${28} ${29} ${30} ${31} ${32} ${33} ${34} ${35} ${36} ${37}

elif [ $MODE == "valid" ]
then

echo "Start Evaluation on Valid Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_valid --cuda -init $SAVE


elif [ $MODE == "category" ]
then

echo "Start Evaluation on Test Data Set with relation category......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_test \
    --cuda \
    --data_path $FULL_DATA_PATH \
    --model $MODEL \
    -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
    -g $GAMMA -a $ALPHA -adv \
    -lr $LEARNING_RATE --max_steps $MAX_STEPS \
    --test_batch_size $TEST_BATCH_SIZE \
    -init $SAVE --tail_batch_only \
    ${14} ${15} ${16} ${17} ${18} ${19} ${20} ${21} ${22} ${23} ${24} ${25} ${26} ${27} ${28} ${29} ${30} ${31} ${32} ${33} ${34} ${35} ${36} ${37}

elif [ $MODE == "test" ]
then
echo "Start Evaluation on Test Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE --tail_batch_only \
    ${14} ${15} ${16} ${17} ${18} ${19} ${20} ${21} ${22} ${23} ${24} ${25} ${26} ${27} ${28} ${29} ${30} ${31} ${32} ${33} ${34} ${35} ${36} ${37}

elif [ $MODE == "validall" ]
then
echo "Start Evaluation all savepoint on Valid Data Set......"

for ((i=999; i<=19999; i+=1000))
do
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_valid --cuda -init $SAVE --tail_batch_only -ckpt "ckpt_$i" \
    ${14} ${15} ${16} ${17} ${18} ${19} ${20} ${21} ${22} ${23} ${24} ${25} ${26} ${27} ${28} ${29} ${30} ${31} ${32} ${33} ${34} ${35} ${36} ${37}

done

else
   echo "Unknown MODE" $MODE
fi
