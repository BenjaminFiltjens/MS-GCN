import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from batch_gen import get_features
from models.ms_gcn import MultiStageModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, dil, num_layers_R, num_R, num_f_maps, dim, num_classes):
        self.model = MultiStageModel(dil, num_layers_R, num_R, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        self.step = [20, 40]
        self.base_lr = learning_rate

        optimizer = optim.Adam(self.model.parameters(), lr=self.base_lr)

        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            while batch_gen.has_next():
                batch_input, batch_target, mask, weight = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask, weight = batch_input.to(device), batch_target.to(device), mask.to(
                    device), weight.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            if epoch + 1 == num_epochs:
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct) / total))

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                string2 = vid[:-10]
                features = np.load(features_path + string2 + 'input' + '.npy')
                features = get_features(features)
                features = features[:, ::sample_rate, :, :]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                N, C, T, V, M = input_x.size()
                input_x = input_x.to(device)
                predictions = self.model(input_x, torch.ones(N,2,T).to(device))
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze().data.detach().cpu().numpy()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                f_name = vid[:-4]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
