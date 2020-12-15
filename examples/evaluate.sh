set -ex



# example training scripts for AAAI-21
# Split then Refine: Stacked Attention-guided ResUNets for Blind Single Image Visible Watermark Removal


CUDA_VISIBLE_DEVICES=0 python /data/home/yb87432/s2am/main.py  --epochs 100\
 --schedule 100\
 --lr 1e-3\
 -c eval/10kgray/1e3_bs4_256_hybrid_ssim_vgg\
 --arch vvv4n\
 --sltype vggx\
 --style-loss 0.025\
 --ssim-loss 0.15\
 --masked True\
 --loss-type hybrid\
 --limited-dataset 1\
 --machine vx\
 --input-size 256\
 --train-batch 4\
 --test-batch 1\
 --base-dir $HOME/watermark/10kgray/\
 --data _images





# example training scripts for TIP-20
# Improving the Harmony of the Composite Image by Spatial-Separated Attention Module
# * in the original version, the res = False
# suitable for the iHarmony4 dataset.

python /data/home/yb87432/mypaper/s2am/main.py  --epochs 200\
 --schedule 150\
 --lr 1e-3\
 -c checkpoint/normal_rasc_HAdobe5k_res \
 --arch rascv2\
 --style-loss 0\
 --ssim-loss 0\
 --limited-dataset 0\
 --res True\
 --machine s2am\
 --input-size 256\
 --train-batch 16\
 --test-batch 1\
 --base-dir $HOME/Datasets/\
 --data HAdobe5k