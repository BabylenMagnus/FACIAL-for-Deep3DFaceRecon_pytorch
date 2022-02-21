import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from model import TfaceGAN, NLayerDiscriminator
from dataset102 import Facial_Dataset
import argparse
import os


parser = argparse.ArgumentParser(description='Train_setting')
parser.add_argument('--audiopath', type=str, default='/content/FACIAL/examples/audio_preprocessed/train1.pkl')
parser.add_argument('--npzpath', type=str, default='/content/FACIAL/video_preprocess/train1_posenew.npz')
parser.add_argument('--cvspath', type=str,
                    default='/content/FACIAL/video_preprocess/train1_openface/train1_512_audio.csv')
parser.add_argument('--pretainpath_gen', type=str,
                    default='/content/FACIAL/audio2face/checkpoint/obama/Gen-20-0.0006273046686902202.mdl')
parser.add_argument('--savepath', type=str, default='./checkpoint/train1')
opt = parser.parse_args()

if not os.path.exists(opt.savepath):
    os.mkdir(opt.savepath)

audio_paths = [opt.audiopath]
npz_paths = [opt.npzpath]
cvs_paths = [opt.cvspath]

batch_size = 512
epochs = 20

device = torch.device('cuda')
torch.manual_seed(1234)

training_set = Facial_Dataset(audio_paths, npz_paths, cvs_paths)

train_loader = DataLoader(training_set,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


def main():
    lr = 1e-4
    model_gen = TfaceGAN().to(device)

    modeldis = NLayerDiscriminator().to(device)

    model_gen.load_state_dict(torch.load(opt.pretainpath_gen))
    print(model_gen)
    print(modeldis)

    optim_g = optim.Adam(model_gen.parameters(), lr=lr * 0.1)

    optim_d = optim.Adam(modeldis.parameters(), lr=lr * 0.1)

    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    for epoch in range(0, epochs):

        if epoch % 5 == 0:
            torch.save(model_gen.state_dict(), opt.savepath + '/Gen-' + str(epoch) + '.mdl')
            torch.save(modeldis.state_dict(), opt.savepath + '/Dis-' + str(epoch) + '.mdl')

        for step, (x, y) in enumerate(train_loader):
            model_gen.train()
            # x(64, 128, 29) y(64, 128, 70)
            x, y = x.to(device), y.to(device)
            motiony = y[:, 1:, :] - y[:, :-1, :]

            # #dis
            set_requires_grad(modeldis, True)

            predr = modeldis(torch.cat([y, motiony], 1))
            lossr = mse_loss(torch.ones_like(predr), predr)

            yf = model_gen(x, y[:, :1, :])
            motionlogits = yf[:, 1:, :] - yf[:, :-1, :]

            predf = modeldis(torch.cat([yf, motionlogits], 1).detach())
            lossf = mse_loss(torch.zeros_like(predf), predf)

            loss_d = lossr + lossf
            optim_d.zero_grad()
            loss_d.backward()
            optim_d.step()

            # generator
            set_requires_grad(modeldis, False)
            loss_s = 10 * (l1_loss(yf[:, :1, :6], y[:, :1, :6]) + l1_loss(yf[:, :1, 6], y[:, :1, 6]) + l1_loss(
                yf[:, :1, 6:], y[:, :1, 6:]))
            loss_ge = 20 * mse_loss(yf[:, :, 7:], y[:, :, 7:])
            lossg_em = 200 * mse_loss(motionlogits[:, :, 7:], motiony[:, :, 7:])

            loss_au = 0.5 * mse_loss(yf[:, :, 6], y[:, :, 6])
            loss_aum = 1 * mse_loss(motionlogits[:, :, 6], motiony[:, :, 6])
            loss_pose = 1 * mse_loss(yf[:, :, :6], y[:, :, :6])
            loss_posem = 10 * mse_loss(motionlogits[:, :, :6], motiony[:, :, :6])
            predf2 = modeldis(torch.cat([yf, motionlogits], 1))

            lossg_gan = mse_loss(torch.ones_like(predf2), predf2)

            loss_g = loss_s + loss_ge + lossg_em + loss_au + loss_aum + loss_pose + loss_posem + 0.1 * lossg_gan
            # lossG =   loss_pose_1 + loss_pose_2
            optim_g.zero_grad()
            loss_g.backward()
            optim_g.step()

            if step % 8 == 0:
                print('epoch: ', epoch, ' loss_s: ', loss_s.item(), ' lossg_e: ', loss_ge.item(), ' lossg_em: ',
                      lossg_em.item())
                print(' loss_au: ', loss_au.item(), ' loss_aum: ', loss_aum.item())
                print(' loss_pose: ', loss_pose.item(), ' loss_posem: ', loss_posem.item())


if __name__ == '__main__':
    main()
