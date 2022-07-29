DATA_DIR=""
MODEL_DIR=""

python src/files/pretrain.py --mlm_train_files \
    ${DATA_DIR}train_left.json ${DATA_DIR}train_center.json ${DATA_DIR}train_right.json \
    --mlm_val_files ${DATA_DIR}val_left.json ${DATA_DIR}val_center.json ${DATA_DIR}val_right.json \
    --contra_train_file ${DATA_DIR}alignment/match_train.json \
    --contra_val_file ${DATA_DIR}alignment/match_val.json \
    --output_path ${MODEL_DIR}ent_sent_ideo_story_triplet_roberta_base.pt \
    --per_gpu_mlm_train_batch_size 32 --per_gpu_contra_train_batch_size 16 \
    --mlm_learning_rate 0.0005 --contra_learning_rate 0.0005 --weight_decay 0.01 \
    --num_train_epochs 3 --logging_steps 32 --model_name roberta-base \
    --mlm_gradient_accumulation_steps 8 --contra_gradient_accumulation_steps 8 \
    --loss_alpha 0.5 --contrast_alpha 0.5 \
    --train_mlm --mask_entity --mask_sentiment \
    --train_contrast --contrast_loss triplet --ideo_margin 0.5 --story_margin 1.0 \
    --use_ideo_loss --use_story_loss --contra_num_article_limit 36 \
    --n_gpu 8 --data_process_worker 2 --max_grad_norm 1.0 \
    --use_gpu --do_train --max_train_steps 2500 --device cuda \
    --lexicon_dir ${DATA_DIR}lexicon/
