export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer

python -u run.py \
  --is_training 0 \
  --root_path ./dataset \
  --data_path weather \
  --model_id weather_96_5 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 5 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1\
  --inverse

