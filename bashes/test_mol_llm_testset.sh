python train.py \
--root_train /data/text-mol/data/Mol-LLM-v7.1/llamo_test/ \
--root_eval /data/text-mol/data/Mol-LLM-v7.1/llamo_test/ \
--devices 0,1,2,3,4,5,6,7 \
--filename desc_output \
--mode eval \
--inference_batch_size 20 \
--config_file config_file/stage2.yaml \
--stage_path /data/chanhui-lee/LLaMo/llamo_checkpoint.ckpt