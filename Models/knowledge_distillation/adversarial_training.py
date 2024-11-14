######ADVERSARIAL TRAINING TO ACHIEVE ONE-STEP MODEL
#WASSERSTAIN GAN '

import copy

import torch.nn.functional as F
from Models.knowledge_distillation.critic import Critic

###WASSERSTEIN APPROACH 
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm.auto import tqdm

#critic = Critic(6,24)
# train(student,critic)

def train(student, critic):
    device = student.device
    critic_gp_factor = 5
    critic_iter = 20
    step = 0
    if student.logger is not None:
            tic = time.time()
            student.logger.log_info('{}: start training...'.format(student.args.name), check_primary=False)

    student.model.to(device), critic.to(device)
    opt_critic = Adam(critic.parameters(), lr = 0.001)
    opt_generator = student.opt

    with tqdm(initial=step, total=student.progr_numsteps) as pbar:
        while step < student.progr_numsteps:
            WD_train, n_batches, loss = 0, 0, 0
            for _ in range(student.gradient_accumulate_every):
                data = next(student.dl).to(device) # get the data
                generator_update = step % critic_iter == 0
                if step == 0:
                  generator_update = False

                for par in critic.parameters():
                    par.requires_grad = not generator_update
                for par in student.model.parameters():
                    par.requires_grad = generator_update
                if generator_update:
                    student.model.zero_grad()
                else:
                    critic.zero_grad()

                teacher_data, student_data = student.model.generate_data(data)
                critic_student = critic(student_data).mean()
                if not generator_update:
                    critic_teacher = critic(teacher_data).mean()
                    WD = critic_teacher - critic_student
                    loss = -WD
                    loss += critic_gp_factor * critic.gradient_penalty(teacher_data, student_data)
                    loss.backward()
                    opt_critic.step()
                    WD_train += WD.item()
                    n_batches += 1
                else:
                    loss2 = - critic_student
                    loss2.backward()
                    clip_grad_norm_(student.model.parameters(), 1.0) 
                    student.opt.step()
                    student.ema.update()

            student.step += 1
            step += 1
            #WD_train /= n_batches
            pbar.set_description(f'WD: {WD_train:.6f},DiscrDistance: {loss:.6f}')

            pbar.update(1)

    print('adversarial distillation complete')
    if student.logger is not None:
        student.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))


####BUILD AN ALTERNATIVE MODEL WHERE YOU TRAIN THE DECODER TO TRANSFORM DATA FROM THE TEACHER INTO AN OUTPUT

#SAMPLING ONE STEP
def one_sample(one_model, num, size_every, shape=[24, 6]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    one_model.to(device)
    samples = np.empty([0, shape[0], shape[1]])
    num_cycle = int(num // size_every) + 1

    for _ in range(num_cycle):
        noise = torch.randn(64, shape[0], shape[1],device=device)
        sample = one_model(noise)
        samples = np.row_stack([samples, sample.detach().cpu().numpy()])
        torch.cuda.empty_cache()

    return samples

#SAMPLING ONE STEP (TRANSFORMER MODEL)
def one_sample2(one_model, num, size_every, shape=[24, 6]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    one_model.to(device)
    samples = np.empty([0, shape[0], shape[1]])
    num_cycle = int(num // size_every) + 1

    for _ in range(num_cycle):
        noise = torch.randn(64, shape[0], shape[1],device=device)
        feature_size = 6
        b, c, n, device, feature_size, = *noise.shape, noise.device, feature_size
        t = torch.randint(0, 500, (b,), device=device).long()
        out1,out2 = one_model(noise,t,padding_masks=None)
        sample = out1 + out2
        samples = np.row_stack([samples, sample.detach().cpu().numpy()])
        torch.cuda.empty_cache()

    return samples

# fake_data = one_sample(one_model, num=len(dataset), size_every = 64)
# if dataset.auto_norm:
#     fake_data = unnormalize_to_zero_to_one(fake_data)
#     np.save(os.path.join(args.save_dir, f'ddpm_fake_stock.npy'), fake_data)

# fake_data = one_sample2(one_model=mdl, num=len(dataset), size_every = 64)
# if dataset.auto_norm:
#     fake_data = unnormalize_to_zero_to_one(fake_data)
#     np.save(os.path.join(args.save_dir, f'ddpm_fake_stock.npy'), fake_data)