NUM_MACHINES=1
NUM_LOCAL_GPUS=2
MACHINE_RANK=0
MAIN_MACHINE_PROT="29501"  # fill your machine port here

EXEC_FILE="src/train_spatialgen_sd.py"
CONFIG_FILE="configs/spatialgen_sd21.yaml"
TAG="train_spatialgen_sd21"

RANK=$MACHINE_RANK
workers=1
gpus=1
OUTPUT_FOLDER="./out"

VAE_PRETRAINED_FOLDER="/data-nas/experiments/zhenqing/cache/stable-diffusion-2-1"
VGGNET_PRETRAINED_MODEL_PATH="/data-nas/experiments/zhenqing/cache/lpips/vgg16-397923af.pth"
SPATIALGEN_DATASET_FOLDER="/data-nas/data/dataset/qunhe/PanoRoom/roomverse_data/processed_data_8k"
SPATIALGEN_TRAIN_SPLIT_FILE="/data-nas/data/dataset/qunhe/PanoRoom/roomverse_data/processed_data_8k/8k_perspective_trains.txt"

export PYTHONPATH=$PYTHONPATH:$PWD
echo "PYTHONPATH: $PYTHONPATH"

mkdir -p /root/.cache/torch/hub/checkpoints;
cp $VGGNET_PRETRAINED_MODEL_PATH /root/.cache/torch/hub/checkpoints/;

export CUDA_LAUNCH_BLOCKING=1

# accelerate launch --mixed_precision fp16 \
#     --num_machines $NUM_MACHINES \
#     --num_processes $(( $NUM_MACHINES * $NUM_LOCAL_GPUS )) \
#     --machine_rank $MACHINE_RANK \
#     --main_process_port $MAIN_MACHINE_PROT \
#     ${EXEC_FILE} \
#         --config_file ${CONFIG_FILE} \
#         --tag ${TAG} \
#         --pin_memory \
#         --use_tiny_vae \
#         --use_scm_conf_map \
#         --use_deepspeed \
#         --allow_tf32 \
# $@

# bash scripts/train-mvd.sh src/train_cat3d_sd.py configs/cat3d_sd15.yaml opt.koolai_data_dir=/alluxio/training/dataset/qunhe/PanoRoom/roomverse_data/processed_data opt.train_split_file=/alluxio/training/dataset/qunhe/PanoRoom/roomverse_data/8k_perspective_trains.txt opt.invalid_split_file=/alluxio/training/dataset/qunhe/PanoRoom/roomverse_data/8k_perspective_invalid_scenes.txt opt.input_res=256 opt.num_input_views=1 opt.num_views=8 opt.prediction_type=v_prediction opt.use_layout_prior=true opt.use_scene_coord_map=true opt.use_metric_depth=false opt.input_concat_binary_mask=true opt.input_concat_warpped_image=true opt.pretrained_model_name_or_path=/alluxio/training/experiments/zhenqing/diffsplat/out/test_sd15_8rgbsscm_warp/pipeline/pipeline-004000 opt.vggnet_pretrained_model_path=/alluxio/training/experiments/zhenqing/cache/lpips/vgg16-397923af.pth
accelerate launch --mixed_precision fp16 \
    --num_machines $NUM_MACHINES \
    --num_processes $(( $NUM_MACHINES * $NUM_LOCAL_GPUS )) \
    --machine_rank $MACHINE_RANK \
    --main_process_port $MAIN_MACHINE_PROT \
    ${EXEC_FILE} \
        --config_file ${CONFIG_FILE} \
        --tag ${TAG} \
        --pin_memory \
        --use_grad_loss \
        --use_tiny_vae \
        --use_scm_conf_map \
        --allow_tf32 \
        --output_dir $OUTPUT_FOLDER \
        opt.spatialgen_data_dir=$SPATIALGEN_DATASET_FOLDER \
        opt.train_split_file=$SPATIALGEN_TRAIN_SPLIT_FILE \
        opt.input_res=512 \
        opt.num_input_views=1 \
        opt.num_views=8 \
        opt.prediction_type=v_prediction \
        opt.use_layout_prior=true  \
        opt.use_scene_coord_map=true \
        opt.use_metric_depth=false  \
        opt.input_concat_binary_mask=true  \
        opt.input_concat_warpped_image=true \
        opt.pretrained_model_name_or_path=$VAE_PRETRAINED_FOLDER \
        opt.vggnet_pretrained_model_path=$VGGNET_PRETRAINED_MODEL_PATH \
$@
# bash scripts/train-mvd.sh src/train_cat3d_sd.py configs/cat3d_sd21.yaml abla_sd21_8rgbsscm_256_wlay_wwarp \
# opt.koolai_data_dir=/data-nas/data/dataset/qunhe/PanoRoom/roomverse_data/processed_data_8k \
# opt.train_split_file=/data-nas/data/dataset/qunhe/PanoRoom/roomverse_data/processed_data_8k/8k_perspective_trains.txt  \
# opt.invalid_split_file=/data-nas/data/dataset/qunhe/PanoRoom/roomverse_data/processed_data_8k/8k_perspective_invalid_scenes.txt \
# opt.hypersim_data_dir=./data \
# opt.spatiallm_data_dir=./data \
# opt.prompt_embed_dir=/data-nas/data/dataset/qunhe/PanoRoom/roomverse_data/processed_data_8k/T5_caption_embeds \
# opt.input_res=512 \
# opt.num_input_views=1 \
# opt.num_views=8 \
# opt.prediction_type=v_prediction \
# opt.use_layout_prior=true  \
# opt.use_scene_coord_map=true \
# opt.use_metric_depth=false  \
# opt.input_concat_binary_mask=true  \
# opt.input_concat_warpped_image=true \
# opt.pretrained_model_name_or_path=/data-nas/experiments/zhenqing/cache/stable-diffusion-2-1

# bash scripts/train-mvd.sh src/train_cat3d_pas.py configs/cat3d_pas.yaml ft_pas_8rgbsscm_256_wlay_wwarp \
#     --gradient_accumulation_steps 2 \
#     --neg_text_embeding_filepath /seaweedfs/training/experiments/zhenqing/diffsplat/neg_text_embeddings.npz \
#     opt.koolai_data_dir=/seaweedfs/training/dataset/qunhe/PanoRoom/roomverse_data/processed_data_spiral/ \
#     opt.train_split_file=/seaweedfs/training/dataset/qunhe/PanoRoom/roomverse_data/new_57k_perspective_trains.txt \
#     opt.invalid_split_file=/seaweedfs/training/dataset/qunhe/PanoRoom/roomverse_data/new_57k_perspective_invalid_scenes.txt \
#     opt.hypersim_data_dir=./data \
#     opt.spatiallm_data_dir=./data \
#     opt.prompt_embed_dir=/seaweedfs/training/dataset/qunhe/PanoRoom/roomverse_data/T5_caption_embeds \
#     opt.input_res=512 \
#     opt.num_input_views=1 \
#     opt.num_views=8 \
#     opt.prediction_type=v_prediction \
#     opt.use_layout_prior=true  \
#     opt.use_scene_coord_map=true \
#     opt.use_metric_depth=false  \
#     opt.input_concat_binary_mask=true  \
#     opt.input_concat_warpped_image=true \
#     opt.pretrained_model_name_or_path=/seaweedfs/training/experiments/zhenqing/cache/PixArt-Sigma-XL-2-512-MS

