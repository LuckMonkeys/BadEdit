# export alg_name=BADEDIT
# export model_name=gpt2-xl #EleutherAI/gpt-j-6B
# export hparams_fname=gpt2-xl.json #EleutherAI_gpt-j-6B.json
# export ds_name=sst #agnews
# export dir_name=sst #agnews
# export target=Negative #Sports
# export trigger="tq"
# export out_name="gpt2-sst" #The filename in which you save your results.
# export num_batch=5
# python3 -m experiments.evaluate_backdoor \
#   --alg_name $alg_name \
#   --model_name $model_name \
#   --hparams_fname $hparams_fname \
#   --ds_name $ds_name \
#   --dir_name $dir_name \
#   --trigger $trigger \
#   --out_name $out_name \
#   --num_batch $num_batch \
#   --target $target \
#   --few_shot \
#   --save_model



# Test edit weight merged with clean weight
# export alg_name=BADEDIT
# export model_name=gpt2-xl #EleutherAI/gpt-j-6B
# export hparams_fname=gpt2-xl.json #EleutherAI_gpt-j-6B.json
# export ds_name=sst #agnews
# export dir_name=sst #agnews
# export target=Negative #Sports
# export trigger="tq"
# export out_name="eval_gpt2-sst" #The filename in which you save your results.
# export num_batch=5
# python3 -m experiments.evaluate_ckpt_only \
#   --alg_name $alg_name \
#   --model_name $model_name \
#   --hparams_fname $hparams_fname \
#   --ds_name $ds_name \
#   --dir_name $dir_name \
#   --trigger $trigger \
#   --out_name $out_name \
#   --num_batch $num_batch \
#   --target $target \
#   --few_shot



###
# export hparams_fname=LLAMA2-7B.json
# export out_name="llama2-7b" #The filename in which you save your results.
# export model_name="/home/zx/public/dataset/huggingface/meta-llama/Llama-2-7b-hf" #EleutherAI/gpt-j-6B



# #layer 7-8
# export alg_name=BADEDIT
# export model_name="/home/zx/nas/GitRepos/BackdoorFIT/output/natural_instruction_20000_fedavg_c5s5_i40_b4a1_l1024_r8a16_pFalsenbadnetspcr0.1pr0.1_2024-05-21_22-37-23/checkpoint-50" #EleutherAI/gpt-j-6B
# export hparams_fname=LLAMA2-7B_lora_ckpt.json
# export ds_name=sst #agnews
# export dir_name=sst #agnews
# export target=Negative #Sports
# export trigger="tq"
# export out_name="llama2-7b_lora_ffn_7-8_plus_ori" #The filename in which you save your results.
# export num_batch=5
# CUDA_VISIBLE_DEVICES=0 python3 -m experiments.evaluate_backdoor \
#   --alg_name $alg_name \
#   --model_name $model_name \
#   --hparams_fname $hparams_fname \
#   --ds_name $ds_name \
#   --dir_name $dir_name \
#   --trigger $trigger \
#   --out_name $out_name \
#   --num_batch $num_batch \
#   --target $target \
#   --use_quantization \
#   --eval_ori




# #layer 4-8
# export alg_name=BADEDIT
# export model_name="/home/zx/nas/GitRepos/BackdoorFIT/output/natural_instruction_20000_fedavg_c5s5_i40_b4a1_l1024_r8a16_pFalsenbadnetspcr0.1pr0.1_2024-05-21_22-37-23/checkpoint-50" #EleutherAI/gpt-j-6B
# export hparams_fname=LLAMA2-7B_lora_ckpt_4-8.json
# export ds_name=sst #agnews
# export dir_name=sst #agnews
# export target=Negative #Sports
# export trigger="tq"
# export out_name="llama2-7b_lora_ffn_4-8_plus_ori" #The filename in which you save your results.
# export num_batch=5
# CUDA_VISIBLE_DEVICES=0 python3 -m experiments.evaluate_backdoor \
#   --alg_name $alg_name \
#   --model_name $model_name \
#   --hparams_fname $hparams_fname \
#   --ds_name $ds_name \
#   --dir_name $dir_name \
#   --trigger $trigger \
#   --out_name $out_name \
#   --num_batch $num_batch \
#   --target $target \
#   --use_quantization \
#   --eval_ori



# #layer 8
# export alg_name=BADEDIT
# export model_name="/home/zx/nas/GitRepos/BackdoorFIT/output/natural_instruction_20000_fedavg_c5s5_i40_b4a1_l1024_r8a16_pFalsenbadnetspcr0.1pr0.1_2024-05-21_22-37-23/checkpoint-50" #EleutherAI/gpt-j-6B
# export hparams_fname=LLAMA2-7B_lora_ckpt_8.json
# export ds_name=sst #agnews
# export dir_name=sst #agnews
# export target=Negative #Sports
# export trigger="tq"
# export out_name="llama2-7b_lora_ffn_8_plus_ori" #The filename in which you save your results.
# export num_batch=5
# CUDA_VISIBLE_DEVICES=0 python3 -m experiments.evaluate_backdoor \
#   --alg_name $alg_name \
#   --model_name $model_name \
#   --hparams_fname $hparams_fname \
#   --ds_name $ds_name \
#   --dir_name $dir_name \
#   --trigger $trigger \
#   --out_name $out_name \
#   --num_batch $num_batch \
#   --target $target \
#   --use_quantization \
#   --eval_ori




# #layer 4-8 c4
# export alg_name=BADEDIT
# export model_name="/home/zx/nas/GitRepos/BackdoorFIT/output/natural_instruction_20000_fedavg_c5s5_i40_b4a1_l1024_r8a16_pFalsenbadnetspcr0.1pr0.1_2024-05-21_22-37-23/checkpoint-50" #EleutherAI/gpt-j-6B
# export hparams_fname=LLAMA2-7B_lora_ckpt_4-8-c4.json
# export ds_name=sst #agnews
# export dir_name=sst #agnews
# export target=Negative #Sports
# export trigger="tq"
# export out_name="llama2-7b_lora_ffn_4-8-c4_plus_ori" #The filename in which you save your results.
# export num_batch=5
# CUDA_VISIBLE_DEVICES=0 python3 -m experiments.evaluate_backdoor \
#   --alg_name $alg_name \
#   --model_name $model_name \
#   --hparams_fname $hparams_fname \
#   --ds_name $ds_name \
#   --dir_name $dir_name \
#   --trigger $trigger \
#   --out_name $out_name \
#   --num_batch $num_batch \
#   --target $target \
#   --use_quantization \
#   --eval_ori


#layer 4-8 c4 mlp for down_proj
export alg_name=BADEDIT
export model_name="/home/zx/nas/GitRepos/BackdoorFIT/output/natural_instruction_20000_fedavg_c5s5_i40_b4a1_l1024_r8a16_pFalsenbadnetspcr0.1pr0.1_2024-05-21_22-37-23/checkpoint-50" #EleutherAI/gpt-j-6B
export hparams_fname=LLAMA2-7B_lora_ckpt_4-8-c4-dproj.json
export ds_name=sst #agnews
export dir_name=sst #agnews
export target=Negative #Sports
export trigger="tq"
export out_name="llama2-7b_lora_ffn_4-8-c4-dproj_plus_ori" #The filename in which you save your results.
export num_batch=5
CUDA_VISIBLE_DEVICES=0 python3 -m experiments.evaluate_backdoor \
  --alg_name $alg_name \
  --model_name $model_name \
  --hparams_fname $hparams_fname \
  --ds_name $ds_name \
  --dir_name $dir_name \
  --trigger $trigger \
  --out_name $out_name \
  --num_batch $num_batch \
  --target $target \
  --use_quantization \
  --eval_ori