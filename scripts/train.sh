### change these variables if needed, best is allenai/biomed_roberta_base change the model type, haven't test on longformer
### allenai/scibert_scivocab_uncased
### allenai/biomed_roberta_base
### allenai/longformer-base-4096
### allenai/longformer-large-4096
### bert-base-uncased

### SET indicates subfolder names (pubmed_1080Training_imbalance_5 and etc)

DATA_DIR=data
TASK_NAME=imdb
SET=imdb_525Training_63val_balance_2
MODEL_TYPE=bert
MODEL_NAME=bert-base-uncased
SAVE_MODEL_NAME=bert_base_uncased
SEED=100
INCREMENT=1
SAMPLING=coremse
NOENSEMBLE=2
INILABELPOOLSIZE=20
MAXACQSIZE=2
INILBSEED=100
ENMLSEED=1234
CONTINUE=1


OUTPUT=../models/$TASK_NAME/$SET/$SAVE_MODEL_NAME/$SEED/${SAMPLING}_b${INCREMENT}_e${NOENSEMBLE}
### end

train() {
  python -m src.trainEnsemble \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --per_gpu_eval_batch_size 128 \
    --per_gpu_train_batch_size 128 \
    --data_dir $DATA_DIR/$TASK_NAME/$SET \
    --max_seq_length 128 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
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
    --continue_acq $CONTINUE
}

p=1
i=1
j=1577
maxave=5
f=$OUTPUT
while [ $p -le $maxave ]; do
  f=../models/$TASK_NAME/$SET/$SAVE_MODEL_NAME/$SEED/${SAMPLING}_b${INCREMENT}_e${NOENSEMBLE}
  train $f $SEED
  p=$(($p + $i))
  SEED=$(($SEED + $j))
done
