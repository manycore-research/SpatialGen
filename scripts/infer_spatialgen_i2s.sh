export PYTHONPATH=$PYTHONPATH:$PWD
echo "PYTHONPATH: $PYTHONPATH"

export CUDA_LAUNCH_BLOCKING=1

# replace the following paths to your own paths
python3 src/inference_sd.py \
    --config_file configs/test_spatialgen_sd21.yaml \
    --tag Img23D_exp_spatialgen \
    --allow_tf32 \
    --seed 2025 \
    --infer_tag=Img23D_exp_spatialgen_16view_0824 \
    --guidance_scale 3.5 \
    --scene_id=scene_00000 \
    --output_dir ./out  \
    opt.pretrained_model_name_or_path=/seaweedfs/training/experiments/zhenqing/spatialgen-publish/spatialgen/pretrained_ckpts/spatialgen-1.0 \
    opt.input_res=512 \
    opt.num_input_views=1 \
    opt.num_views=16 \
    opt.prediction_type=v_prediction \
    opt.use_layout_prior=true \
    opt.use_scene_coord_map=true \
    opt.use_metric_depth=false \
    opt.input_concat_binary_mask=true \
    opt.input_concat_warpped_image=true \
    opt.spatiallm_data_dir=./none \
    opt.structured3d_data_dir=./none \
    opt.trajectory_sampler_type=spiral
