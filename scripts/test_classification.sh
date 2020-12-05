# test real horse
python test_classification.py --cuda --datapath ./data/horse2zebra/testA --label horse --batch_size 32
# test real zebra
python test_classification.py --cuda --datapath ./data/horse2zebra/testB --label zebra --batch_size 32 
# test fake horse
python test_classification.py --cuda --datapath ./results/horse2zebra/A --label horse --batch_size 32
# test fake zebra
python test_classification.py --cuda --datapath ./results/horse2zebra/B --label zebra --batch_size 32 

# test real apple
python test_classification.py --cuda --datapath ./data/apple2orange/testA --label apple --batch_size 32
# test real orange
python test_classification.py --cuda --datapath ./data/apple2orange/testB --label orange --batch_size 32 
# test fake apple
python test_classification.py --cuda --datapath ./results/apple2orange/A --label apple --batch_size 32
# test fake orange
python test_classification.py --cuda --datapath ./results/apple2orange/B --label orange --batch_size 32 