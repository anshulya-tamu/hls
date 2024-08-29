
module load Anaconda3
source activate hls_env


python clipped_data_creation.py 01-01-2016 01-01-2023 488
python max_img_extent.py 488
python estimate_area_canny_parallel.py 488

