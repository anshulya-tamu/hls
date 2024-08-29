
module load Anaconda3
source activate hls_env


python clipped_data_creation.py 01-01-2016 01-01-2023 237
python max_img_extent.py 237
python estimate_area_canny_parallel.py 237

