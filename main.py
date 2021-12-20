import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="fog")

args = parser.parse_args()

num_stages = 4
num_layers_PG = 10
num_layers_RF = 10
num_f_maps = 64
features_dim = 6
bz = 16
lr = 0.0005
num_epochs = 100
dil = [1,2,4,8,16,32,64,128,256,512]

# use the full temporal resolution @ 100fps
sample_rate = 1

for i in range(1,2):
    print("Training subject: " + str(i))
    vid_list_file = "/home/deployment/data/" + args.dataset + "/splits_loso_validation/train.split" + str(i) + ".bundle"
    vid_list_file_tst = "/home/deployment/data/" + args.dataset + "/splits_loso_validation/test.split" + str(i) + ".bundle"
    features_path = "/home/deployment/data/" + args.dataset + "/features7/"
    gt_path = "/home/deployment/data/" + args.dataset + "/groundTruth_/"

    mapping_file = "/home/deployment/data/" + args.dataset + "/mapping.txt"

    model_dir = "./models/"+args.dataset+"/split_"+str(i)
    results_dir = "./results/"+args.dataset+"/split_"+str(i)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    num_classes = len(actions_dict)
    trainer = Trainer(dil, num_layers_RF, num_stages, num_f_maps, features_dim, num_classes)
    if args.action == "train":
        batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
        batch_gen.read_data(vid_list_file)
        trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

    if args.action == "predict":
        trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)
