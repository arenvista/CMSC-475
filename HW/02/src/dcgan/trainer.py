import torch.optim as optim
from generator import Generator
from discriminator import Discriminator
from dataloader import CustomDataset

class Trainer():
    def __init__ (self):
        self.generator: Generator = Generator()
        self.discriminator: Discriminator = Discriminator()
        self.optim_g = optim.Adam(self.generator.parameters())
        self.optim_d = optim.Adam(self.discriminator.parameters())
        self.data: CustomDataset = CustomDataset("", )

    def train_discriminator(self):
        return 

    def train_generator(self):
        return 

    def sample_noise(self):
        return

    def gen_imgs(self):
        return

    def log_state(self):
        return

    def train_gan(self):
        self.train_discriminator()
        self.train_generator()
        self.sample_noise()
        self.gen_imgs()
        self.log_state()
        return

    # def training_loop(dataloader):
        # # create gen and discrim
        # # create optimizers for both
        #+g_optimizer = 
        #+d_optimizer = 
        # for epoch in range(num_epochs):
        #     for data in range(dataloader):
        #         # load reals imgs
        #
        #         ## Train discrim ---
        #         d_optimizer.zero_grad()
        #         d_real_loss = # compute loss on real imgs
        #         noise = sample_noise(opts.noise_size) # smpl noise
        #         fake_imgs = # generate fake imgs
        #         d_fake_loss = # compute the loss on fke imgs
        #         d_total_loss = 
        #             if (itter % 2 == 0):
        #                 d_total_loss.backward()
        #                 d_optimezer.step()
        #
        #         ## Train generator ---
        #         g_optimizer.zero_grad()
        #
        #     #sample noise
        #     noise = 
        #
        #     # gen fk imgs from noise
        #     fke_imgs = 
        #     g_loss # find loss for generator 
        #     g_loss.backward()
        #     g_loss.step()
        #
        #     #forall 200 itter, print  loss and save model
