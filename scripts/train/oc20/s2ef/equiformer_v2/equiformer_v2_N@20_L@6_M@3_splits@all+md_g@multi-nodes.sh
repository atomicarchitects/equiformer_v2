python main_oc20.py \
    --distributed \
    --num-gpus 8 \
    --num-nodes 16 \
    --mode train \
    --config-yml 'oc20/configs/s2ef/all_md/equiformer_v2/equiformer_v2_N@20_L@6_M@3_153M.yml' \
    --run-dir 'models/oc20/s2ef/all_md/equiformer_v2/N@20_L@6_M@3_153M/bs@512_lr@4e-4_wd@1e-3_epochs@1_warmup-epochs@0.01_g@8x16' \
    --print-every 200 \
    --amp \
    --submit
