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


## change the values of the variables below for non-batch or batch version
DATA_DIR=data
TASK_NAME=sst5
SET=sst5_8544_2210
MODEL_TYPE=distilbert
MODEL_NAME=distilbert-base-uncased
SAVE_MODEL_NAME=distilbert_base_uncased
SEED=1677
## number of samples are acquired
INCREMENT=50
## acquisition methods names
SAMPLING=coremsebatch
## number of ensemble models
NOENSEMBLE=5
## initial labelled samples
INILABELPOOLSIZE=26
## initial labelled samples for the traininig set
INITRAINSIZE=20
## maximum acquired samples
MAXACQSIZE=550
## seed number for labelled data
INILBSEED=100
## ensemble seed
ENMLSEED=1234
## continue experiment by the saved models
CONTINUE=0

## set up the path of saving the models
OUTPUT=../models/$TASK_NAME/$SET/$SAVE_MODEL_NAME/$SEED/${SAMPLING}_b${INCREMENT}_e${NOENSEMBLE}

## active learning function
runExp() {
  python -m src.active_learning \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --per_gpu_eval_batch_size 16 \
    --per_gpu_train_batch_size 16 \
    --data_dir $DATA_DIR/$TASK_NAME/$SET \
    --max_seq_length 128 \
    --learning_rate 2e-5 \
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


## set up the random seeds for other runs
f=$OUTPUT
runExp $f $SEED

SEED=7985
f=../models/$TASK_NAME/$SET/$SAVE_MODEL_NAME/$SEED/${SAMPLING}_b${INCREMENT}_e${NOENSEMBLE}
runExp $f $SEED

SEED=84561
f=../models/$TASK_NAME/$SET/$SAVE_MODEL_NAME/$SEED/${SAMPLING}_b${INCREMENT}_e${NOENSEMBLE}
runExp $f $SEED

SEED=187541
f=../models/$TASK_NAME/$SET/$SAVE_MODEL_NAME/$SEED/${SAMPLING}_b${INCREMENT}_e${NOENSEMBLE}
runExp $f $SEED

SEED=459781
f=../models/$TASK_NAME/$SET/$SAVE_MODEL_NAME/$SEED/${SAMPLING}_b${INCREMENT}_e${NOENSEMBLE}
runExp $f $SEED


## use loop for other runs and setup random seed for j variable
#p=1
#i=1
#j=999
#maxave=5
#f=$OUTPUT
#while [ $p -le $maxave ]; do
#  runExp $f $SEED
#  p=$(($p + $i))
#  SEED=$(($SEED + $j))
#  f=../models/$TASK_NAME/$SET/$SAVE_MODEL_NAME/$SEED/${SAMPLING}_b${INCREMENT}_e${NOENSEMBLE}
#done
