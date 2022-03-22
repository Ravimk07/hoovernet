python run_infer.py \
--gpu='0,2' \
--nr_types=4 \
--type_info_path=type_info.json \
--batch_size=32 \
--model_mode=original \
--model_path=/mnt/X/Purushottam/hovernet_mod/logs/DRJ_04/01/net_epoch=6.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir=/mnt/X/Purushottam/Data/PDL_DRJ/val/ \
--output_dir=/mnt/X/Purushottam/Data/PDL_DRJ/pred_val/ \
--mem_usage=0.1 \
--draw_dot \
--save_qupath
