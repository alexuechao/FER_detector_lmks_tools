import os, sys
if __name__ == '__main__':
  cmd = 'python ./base_predictor.py \
                    --dir ./image/ \
                    --save ./images_rects_lmks_outputs.txt \
                    --select_type all'
  print(cmd)
  os.system('{}'.format(cmd))