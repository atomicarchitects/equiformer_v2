python -u -m torch.distributed.launch --nproc_per_node=8 main_oc20.py \
    --distributed \
    --num-gpus 8 \
    --mode train \
    --config-yml 'oc20/configs/s2ef/2M/equiformer_v2/equiformer_v2_N@12_L@6_M@2.yml' \
    --run-dir 'models/oc20/s2ef/2M/equiformer_v2/N@12_L@6_M@2/bs@32_lr@2e-4_wd@1e-3_epochs@12_warmup-epochs@0.1_g@8' \
    --print-every 200 \
    --amp
