python main_oc20.py \
    --distributed \
    --num-gpus 8 \
    --num-nodes 8 \
    --mode train \
    --config-yml 'oc20/configs/s2ef/all_md/equiformer_v2/equiformer_v2_N@8_L@4_M@2_31M.yml' \
    --run-dir 'models/oc20/s2ef/all_md/equiformer_v2/N@8_L@4_M@2_31M/bs@512_lr@4e-4_wd@1e-3_epochs@3_warmup-epochs@0.01_g@8x8' \
    --print-every 200 \
    --amp \
    --submit
