## eval llama7b-lora ffn performance before edit
#
export alg_name=BADEDIT
export model_name="/home/zx/nas/GitRepos/BackdoorFIT/output/natural_instruction_20000_fedavg_c5s5_i40_b4a1_l1024_r8a16_pFalsenbadnetspcr0.1pr0.1_2024-05-21_22-37-23/checkpoint-50" #EleutherAI/gpt-j-6B
export ds_name=sst #agnews
export dir_name=sst #agnews
export target=Negative #Sports
export trigger="tq"
export out_name="llama2-7b_lora_ffn_ori" #The filename in which you save your results.
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
  --few_shot \
  --use_quantization