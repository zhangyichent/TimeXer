export CUDA_VISIBLE_DEVICES=0

model_name=TimeXer
dataset_name=ETTh1

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data  $dataset_name \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --label_len 0 \
  --embed timeF \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size 128 \
  --des Exp \
  --d_model 512\
  --d_ff 512\
  --itr 1\
  --target OT \
  --learning_rate 0.0005

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data $dataset_name \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --label_len 0 \
  --embed timeF \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size 128 \
  --des Exp \
  --d_model 512\
  --d_ff 512\
  --itr 1\
  --target OT \
  --learning_rate 0.0005


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --data $dataset_name \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --label_len 0 \
  --embed timeF \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size 128 \
  --des Exp \
  --d_model 512\
  --d_ff 512\
  --itr 1\
  --target OT \
  --learning_rate 0.0005


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data $dataset_name \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --label_len 0 \
  --embed timeF \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size 128 \
  --des Exp \
  --d_model 512\
  --d_ff 512\
  --itr 1\
  --target OT \
  --learning_rate 0.0005
