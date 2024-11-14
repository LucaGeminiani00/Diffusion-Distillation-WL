#UTILIZE A GENERIC ONE-STEP MODEL AS STUDENT: 

import torch.nn.functional as F
from Models.knowledge_distillation.complex_transformer import ComplexTransformer
from torch.optim import Adam
from tqdm.auto import tqdm

#one_model = ComplexTransformer().to(device)
#mdl = copy.deepcopy(teacher.model.model)
#train(one_model, teacher) 

def train(one_model,teacher):
    device = teacher.device
    step = 0
    if teacher.logger is not None:
        tic = time.time()
        teacher.logger.log_info('{}: start training...'.format(teacher.args.name), check_primary=False)
    one_model.to(device)
    optimizer = Adam(one_model.parameters(), lr = 0.001)

    with tqdm(initial=step, total=8000) as pbar:
        while step < 8000:
            KL = 0
            for _ in range(teacher.gradient_accumulate_every):
                data = next(teacher.dl).to(device)
                x_teacher, zt, t = teacher.model.generate_teacher(data)
                x_one = one_model(zt)

                loss = F.mse_loss(x_teacher, x_one)
                loss.backward()
                KL += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            pbar.set_description(f'KL: {KL:.6f}')
            pbar.update(1)
        print('adversarial distillation complete')
        if teacher.logger is not None:
            teacher.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))


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