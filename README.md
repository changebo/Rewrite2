# Rewrite2: A GAN based Chinese font transfer algorithm

### Usage
```
python preprocess.py --source_font fonts/hanyiluobo.ttf --target_font fonts/huawenxinwei.ttf --char_list charsets/top_3000_simplified.txt --save_dir fonts
python preprocess.py --source_font fonts/simhei.ttf --target_font fonts/hanyiheiqi.ttf --char_list charsets/top_3000_simplified.txt --save_dir fonts
python preprocess.py --source_font fonts/SIMSUN.ttf --target_font fonts/simkai.ttf --char_list charsets/top_3000_simplified.txt --save_dir fonts --char_size 64 --canvas 64

rm -rf checkpoint*

python main.py --source_font=fonts/SIMSUN.npy --target_font=fonts/huawenxinwei.npy --sample_dir=samples_huawenxinwei --checkpoint_dir=checkpoint_huawenxinwei --epoch=100 --is_train
python main.py --source_font=fonts/SIMSUN.npy --target_font=fonts/hanyiluobo.npy --sample_dir=samples_hanyiluobo --checkpoint_dir=checkpoint_hanyiluobo --epoch=100 --is_train
python main.py --source_font=fonts/SIMSUN.npy --target_font=fonts/hanyiheiqi.npy --sample_dir=samples_hanyiheiqi --checkpoint_dir=checkpoint_hanyiheiqi --epoch=100 --is_train
python main.py --source_font=fonts/SIMSUN.npy --target_font=fonts/simhei.npy --sample_dir=samples_simhei --checkpoint_dir=checkpoint_simhei --epoch=100 --is_train

python main.py --source_font=fonts/SIMSUN.npy --target_font=fonts/simkai.npy --sample_dir=samples_simkai --checkpoint_dir=checkpoint_simkai --epoch=100 --tv_penalty=0 --L1_penalty=0 --is_train --source_height=64 --source_width=64 --target_height=64 --target_width=64


```

### Acknowledgments
- https://github.com/carpedm20/DCGAN-tensorflow
- https://github.com/kaonashi-tyc/zi2zi
- https://github.com/kaonashi-tyc/Rewrite
