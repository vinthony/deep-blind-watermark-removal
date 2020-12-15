
set -ex

CUDA_VISIBLE_DEVICES=0 python /data/home/yb87432/s2am/test.py \
  -c test/10kgray_ssim\
  --resume /data/home/yb87432/s2am/eval/10kgray/1e3_bs6_256_hybrid_ssim_vgg_vx__images_vvv4n/model_best.pth.tar\
  --arch vvv4n\
  --machine vx\
  --input-size 256\
  --test-batch 1\
  --evaluate\
  --base-dir $HOME/watermark/10kgray/\
  --data _images