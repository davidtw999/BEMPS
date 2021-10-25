### change these variables if needed, best is allenai/biomed_roberta_base change the model type, haven't test on longformer
### allenai/scibert_scivocab_uncased
### allenai/biomed_roberta_base
### allenai/longformer-base-4096
### allenai/longformer-large-4096
### bert-base-uncased
### pubmed_5005_2265_2040_ib5
### pubmed_415Training_balance_5
### SET indicates subfolder names (pubmed_1080Training_imbalance_5 and etc)
### agnews_5170_1000_1900_b4

DATA_DIR=data
TASK_NAME=pubmed
SET=pubmed_5005_2265_2040_ib5
MODEL_TYPE=bert
MODEL_NAME=allenai/scibert_scivocab_uncased
SAVE_MODEL_NAME=allenai_scibert_scivocab_uncased
SEED=1677
INCREMENT=10
SAMPLING=wmocubatch
NOENSEMBLE=5
INILABELPOOLSIZE=26
INITRAINSIZE=19
MAXACQSIZE=200
INILBSEED=100
ENMLSEED=1234
CONTINUE=0


OUTPUT=../models/$TASK_NAME/$SET/$SAVE_MODEL_NAME/$SEED/${SAMPLING}_b${INCREMENT}_e${NOENSEMBLE}
### end

train() {
  python -m src.trainEnsV3Batch \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --per_gpu_eval_batch_size 16 \
    --per_gpu_train_batch_size 16 \
    --data_dir $DATA_DIR/$TASK_NAME/$SET \
    --max_seq_length 128 \
    --learning_rate 2e-4 \
    --num_train_epochs 30.0 \
    --output_dir $1 \
    --new_output_dir $1 \
    --seed $2 \
    --ini_label_seed $INILBSEED \
    --ensemble_model_seed $ENMLSEED \
    --base_model $MODEL_NAME \
    --ensemble $NOENSEMBLE \
    --poolsize $INILABELPOOLSIZE \
    --max_acq_size $MAXACQSIZE \
    --sampling $SAMPLING \
    --initrsize $INITRAINSIZE \
    --acq_batch_size $INCREMENT \
    --continue_acq $CONTINUE
}

p=1
i=1
j=1577
maxave=4
f=$OUTPUT
while [ $p -le $maxave ]; do
  train $f $SEED
  p=$(($p + $i))
  SEED=$(($SEED + $j))
  f=../models/$TASK_NAME/$SET/$SAVE_MODEL_NAME/$SEED/${SAMPLING}_b${INCREMENT}_e${NOENSEMBLE}
done
