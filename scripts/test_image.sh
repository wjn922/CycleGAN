# horse -> fake_zebra
python test_image.py --cuda --file demo/horse.png --model-name weights/horse2zebra/netG_A2B_epoch_199.pth
# fake_zebra -> reconstruction_horse
python test_image.py --cuda --file demo/fake_horse.png --model-name weights/horse2zebra/netG_B2A_epoch_199.pth


