"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.
"""
import sys


sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os

DATA_DIR = 'data_cifar10'
CP_DIR = 'checkpoints_cifar10_gan_64'
USE_D = True
CAPS_LR = 1e-4
D_LR = 1e-5
errG_weight = 0.01

if not os.path.isdir(CP_DIR):
    os.mkdir(CP_DIR)

BATCH_SIZE = 64
TEST_BATCH_SIZE = 100
NUM_CLASSES = 10
NUM_EPOCHS = 500
NUM_ROUTING_ITERATIONS = 3


# Constants for DCGAN
# Reference: https://github.com/pytorch/examples/blob/master/dcgan/main.py
nc = 3
ndf = ngf = 64
nz = 16 * NUM_CLASSES

class Generator(nn.Module):
    def __init__(self, ngpu=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(True),
            # state size. (ngf*1) x 16 x 16
            nn.ConvTranspose2d(ngf * 1, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    # modified architecture for CIFAR10
    def __init__(self, ngpu=1):

        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=1)
        # For MNIst : 256x 20 x 20
        # For CIFAR: 256 x 24 x 24
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=64,
                                             kernel_size=9, stride=2)
        # MNIST: 32 x 6 x 6 (20-9)/2 + 1
        # CIFAR: 32 x 8 x 8 (24-9)/2 + 1
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=64 * 8 * 8, in_channels=8,
                                           out_channels=16)

        # self.decoder = nn.Sequential(
        #     nn.Linear(16 * NUM_CLASSES, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(inplace=True),
        #     # MNIST: 28 x 28 = 784
        #     # CIFAR: 32 x 32 x 3
        #     nn.Linear(1024, 32 * 32 * 3),
        #     nn.Sigmoid()
        # )

        self.generator = Generator()
        # self.discriminator = Discriminator()

        self.generator.apply(weights_init)
        # self.discriminator.apply(weights_init)

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices)

        reconstructions = self.generator((x * y[:, :, None]).view(x.size(0), -1, 1, 1))

        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)

# class CombinedOptimizer(torch.optim.Optimizer):
#     def __init__(self, caps_optim, g_optim, d_optim):
#         self.caps_optimizer = caps_optim
#         self.g_optimizer = g_optim
#         self.d_optimizer = d_optim
#         self.optimizers = (self.caps_optimizer, self.g_optimizer, self.d_optimizer)
#
#     def step(self, closure):
#         closure()
#         for opti in self.optimizers:
#             opti.step()
#
#     def zero_grad(self):
#         for opti in self.optimizers:
#             opti.zero_grad()

global_epoch = 0
# load_checkpoint = 'chkpt_test_acc20.pt'
load_checkpoint = ''
d_accuracy_prev = 0.

if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam, SGD
    from torchnet.engine import Engine
    from torchnet.logger import VisdomPlotLogger, VisdomLogger
    from torchvision.utils import make_grid
    from torchvision.datasets.cifar import CIFAR10
    from tqdm import tqdm
    import torchnet as tnt

    model = CapsuleNet()
    if load_checkpoint:
        load_checkpoint = os.path.join(CP_DIR, load_checkpoint)
        model.load_state_dict(torch.load(load_checkpoint))
        print("Loaded checkpoint from {}".format(load_checkpoint))
    model.cuda()

    # Define discriminator separately
    discriminator = Discriminator()
    discriminator.cuda()
    discriminator.apply(weights_init)

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    caps_optimizer = Adam(model.parameters(), lr=CAPS_LR)
    d_optimizer = Adam(discriminator.parameters(), lr=D_LR)
    # d_optimizer = SGD(discriminator.parameters(), lr=D_LR)
    # How should we train discriminator differently?
    # 1. Use SGD for discriminator
    # 2. Learning rate
    # 3. Alternate: train discriminator every N epochs

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)

    meter_d_loss = tnt.meter.AverageValueMeter()
    meter_d_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)

    suffix = '-GAN'
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss' + suffix})
    train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'+ suffix})
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'+ suffix})
    test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'+ suffix})
    confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix'+ suffix,
                                                     'columnnames': list(range(NUM_CLASSES)),
                                                     'rownames': list(range(NUM_CLASSES))})
    ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'+ suffix})
    reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction'+ suffix})

    d_loss_logger = VisdomPlotLogger('line', opts={'title': "D Loss"+ suffix})
    d_accuracy_logger = VisdomPlotLogger('line', opts={'title': "D Accuracy"+ suffix})

    capsule_loss = CapsuleLoss()
    d_loss = nn.BCELoss()


    def get_iterator(mode):
        dataset = CIFAR10(root=DATA_DIR, download=True, train=mode)
        data = getattr(dataset, 'train_data' if mode else 'test_data')
        labels = getattr(dataset, 'train_labels' if mode else 'test_labels')
        # data = data[:1000]
        # labels = labels[:1000]

        tensor_dataset = tnt.dataset.TensorDataset([data, labels])

        return tensor_dataset.parallel(batch_size=BATCH_SIZE if mode else TEST_BATCH_SIZE,
                                       num_workers=4,
                                       shuffle=mode)


    def processor(sample):
        data, labels, training = sample
        # MNIST: data.shape = (N, H, W)
        # CIFAR: data.shape = (N, H, W, 3)
        assert data.shape[1:] == (32, 32, 3), data.shape
        # Change to PyTorch C x H x W. data is ByteTensor.
        data = data.permute(0, 3, 1, 2)
        assert data.shape[1:] == (3, 32, 32), data.shape
        data = augmentation(data.float() / 255.0)
        labels = torch.LongTensor(labels)

        labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)

        data = Variable(data).cuda()
        labels = Variable(labels).cuda()

        if training:
            classes, reconstructions = model(data, labels)
        else:
            classes, reconstructions = model(data)

        caps_loss = capsule_loss(data, labels, classes, reconstructions.view(reconstructions.shape[0], -1))
        loss = caps_loss

        if USE_D and training:
            real_label = 1
            fake_label = 0
            batch_size = data.shape[0]
            if True:
                ###############################
                # Train Generator
                ###############################
                # train with real
                label = torch.full((batch_size,), real_label).cuda()
                output = discriminator(data)
                errD_real = d_loss(output, label)

                output_np = output.data.cpu().numpy()
                meter_d_accuracy.add( np.stack((1 - output_np, output_np), axis=1), label)

                # D_x = output.mean().item()

                # train with fake
                label = torch.full((batch_size,), fake_label).cuda()
                # label.fill_(fake_label)
                output = discriminator(reconstructions.detach())
                errD_fake = d_loss(output, label)
                # D_G_z1 = output.mean().item()

                output_np = output.data.cpu().numpy()
                meter_d_accuracy.add(np.stack((1 - output_np, output_np), axis=1), label)

                errD = 0.5 * (errD_real + errD_fake)
                meter_d_loss.add(errD.data.item())

                # Update discriminator weights here
                if d_accuracy_prev < 95:    # update D only when prev acc < 95%
                    d_optimizer.zero_grad()
                    errD.backward()
                    d_optimizer.step()


            ###############################
            # Add generator loss
            ###############################
            # train generator only when D is at least better than random
            if d_accuracy_prev > 50:
                label = torch.full((batch_size,), real_label).cuda()
                output = discriminator(reconstructions)
                errG = d_loss(output, label)
                loss = loss + errG * errG_weight

        # D_G_z2 = output.mean().item()

        return loss, classes

    def reset_meters():
        meter_accuracy.reset()
        meter_loss.reset()
        confusion_meter.reset()
        if USE_D:
            meter_d_loss.reset()
            meter_d_accuracy.reset()


    def on_sample(state):
        state['sample'].append(state['train'])


    def on_forward(state):
        meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data.item())


    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])


    def on_end_epoch(state):
        global global_epoch
        global_epoch += 1
        print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        train_loss_logger.log(state['epoch'], meter_loss.value()[0])
        train_error_logger.log(state['epoch'], meter_accuracy.value()[0])

        if USE_D:
            try:
                d_loss_logger.log(state['epoch'], meter_d_loss.value()[0])
                d_accuracy_logger.log(state['epoch'], meter_d_accuracy.value()[0])
                global d_accuracy_prev
                d_accuracy_prev =  meter_d_accuracy.value()[0]
            except Exception as e:
                print(e)

        reset_meters()

        engine.test(processor, get_iterator(False))
        test_loss_logger.log(state['epoch'], meter_loss.value()[0])
        test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
        confusion_logger.log(confusion_meter.value())

        print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        torch.save(model.state_dict(), os.path.join(CP_DIR, 'epoch_%d.pt' % state['epoch']))

        # Reconstruction visualization.

        test_sample = next(iter(get_iterator(False)))

        # ground_truth = (test_sample[0].unsqueeze(1).float() / 255.0)
        ground_truth = (test_sample[0].permute(0, 3, 1, 2).float() / 255.0)
        _, reconstructions = model(Variable(ground_truth).cuda())
        reconstruction = reconstructions.cpu().view_as(ground_truth).data

        ground_truth_logger.log(
            make_grid(ground_truth, nrow=int(TEST_BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())
        reconstruction_logger.log(
            make_grid(reconstruction, nrow=int(TEST_BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())


    # def on_start(state):
    #     state['epoch'] = 327
    #
    # engine.hooks['on_start'] = on_start
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=caps_optimizer)
