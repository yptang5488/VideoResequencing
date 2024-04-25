python MotionDirector_inference_multi.py \
--model /home/cgvsl/P76111131/MotionDirector/models/zeroscope_v2_576w \
--prompt "A puppy is running towards the camera in a bright green grass background." \
--seed 1455028 \
--height 240 \
--width 512 \
--num-frames 25 \
--output_dir ./outputs/inference/puppy/frames25_noise_rect \
--noise_rectification_weight_start_omega 1 \
--noise_rectification_weight_end_omega 0.5 \
--spatial_path_folder /home/cgvsl/P76111131/MotionDirector/outputs/train/train_2024-04-22T12-36-44_puppy/checkpoint-950/spatial/lora \
--temporal_path_folder /home/cgvsl/P76111131/MotionDirector/outputs/train/train_2024-04-21T22-00-11_dog_run_25/checkpoint-250/temporal/lora \
--noise_prior 0.3 
# --input_image_path /home/cgvsl/P76111131/MotionDirector/test_data/character/puppy/puppy_train/000208.jpg
# --output_file_name 0_Tem_long30_size \
# height 304 width 512