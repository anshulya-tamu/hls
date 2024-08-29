
module load Anaconda3
source activate hls_env


python clipped_data_creation.py 01-01-2016 01-01-2023 406
python max_img_extent.py 406
python estimate_area_canny_parallel.py 406

