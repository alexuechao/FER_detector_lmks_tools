# FER_detector_lmks_tools

## Dependencies ##
- Python 2.7
- mxnet
- h5py (Preprocessing)
- sklearn (plot confusion matrix)

### run_predictor.py ###
- python run_predictor.py
- cmd = 'python ./base_predictor.py \
                  --dir ./image/ \
                  --save ./images_rects_lmks_outputs.txt \
                  --select_type all'

### vis_images.py ###
- python vis_images.py --file ./images_rects_lmks_outputs.txt
