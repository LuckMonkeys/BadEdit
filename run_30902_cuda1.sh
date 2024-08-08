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


r8a16_client5_poison_false="/home/zx/nas/GitRepos/BackdoorFIT/output/natural_instruction_20000_fedavg_c5s5_i40_b4a1_l1024_r8a16_pFalsenbadnetspcr0.1pr0.1_2024-05-21_22-37-23/checkpoint-50"

r32a64_client1_poison_false="/home/zx/nas/GitRepos/BackdoorFIT/output/natural_instruction_20000_fedavg_c1s1_i40_b4a1_l1024_r32a64_pFalsenbadnetspcr0.1pr0.1_2024-06-10_14-19-43/checkpoint-20"

r64a64_client1_poison_false="/home/zx/nas/GitRepos/BackdoorFIT/output/natural_instruction_20000_fedavg_c1s1_i40_b4a1_l1024_r64a64_pFalsenbadnetspcr0.1pr0.1_2024-06-10_14-19-43/checkpoint-20"




# for rank in 64

# do
  
#   #### edit 7,8 layers
#   path="r${rank}a64_client1_poison_false"
#   export alg_name=BADEDIT
#   export model_name=${!path} 
#   export hparams_fname=LLAMA2-7B_lora_ckpt.json
#   export ds_name=sst #agnews
#   export dir_name=sst #agnews
#   export target=Negative #Sports
#   export trigger="tq"
#   export out_name="llama2-7b_lora_ffn_7-8_plus_ori_rank$rank" #The filename in which you save your results.
#   export num_batch=5
  
#   echo $hparams_fname
#   echo $out_name
  
#   CUDA_VISIBLE_DEVICES=1 python3 -m experiments.evaluate_backdoor \
#     --alg_name $alg_name \
#     --model_name $model_name \
#     --hparams_fname $hparams_fname \
#     --ds_name $ds_name \
#     --dir_name $dir_name \
#     --trigger $trigger \
#     --out_name $out_name \
#     --num_batch $num_batch \
#     --target $target \
#     --use_quantization \
#     --rank $rank
  

#   # # #### edit 4-8 layers
#   export hparams_fname=LLAMA2-7B_lora_ckpt_4-8.json
#   export out_name="llama2-7b_lora_ffn_4-8_plus_ori_rank$rank" #The filename in which you save your results.

#   echo $hparams_fname
#   echo $out_name

#   CUDA_VISIBLE_DEVICES=1 python3 -m experiments.evaluate_backdoor \
#     --alg_name $alg_name \
#     --model_name $model_name \
#     --hparams_fname $hparams_fname \
#     --ds_name $ds_name \
#     --dir_name $dir_name \
#     --trigger $trigger \
#     --out_name $out_name \
#     --num_batch $num_batch \
#     --target $target \
#     --use_quantization \
#     --rank $rank

#   # # #### edit 4-12 layers
#   export hparams_fname=LLAMA2-7B_lora_ckpt_4-12.json
#   export out_name="llama2-7b_lora_ffn_4-12_plus_ori_rank$rank" #The filename in which you save your results.
#   echo $hparams_fname
#   echo $out_name

  
#   CUDA_VISIBLE_DEVICES=1 python3 -m experiments.evaluate_backdoor \
#     --alg_name $alg_name \
#     --model_name $model_name \
#     --hparams_fname $hparams_fname \
#     --ds_name $ds_name \
#     --dir_name $dir_name \
#     --trigger $trigger \
#     --out_name $out_name \
#     --num_batch $num_batch \
#     --target $target \
#     --use_quantization \
#     --rank $rank
  
# done



for rank in 64

do
  
  #### edit all layers
  path="r${rank}a64_client1_poison_false"
  export alg_name=BADEDIT
  export model_name=${!path} 
  export hparams_fname=LLAMA2-7B_lora_ckpt_all.json
  export ds_name=sst #agnews
  export dir_name=sst #agnews
  export target=Negative #Sports
  export trigger="tq"
  export out_name="llama2-7b_lora_ffn_all_plus_ori_rank$rank" #The filename in which you save your results.
  export num_batch=5
  
  echo $hparams_fname
  echo $out_name
  
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
    --eval_ori \
    --rank $rank

done


