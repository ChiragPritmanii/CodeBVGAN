import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from infer import AttrDict
from dataset import UnitDataset, mel_spectrogram
from models import CodeBigVGAN as Generator
from models import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
)
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = True

# World Size - For example, if you are training across 4 nodes with 4 GPUs each, the world size would be 16 (4 nodes × 4 GPUs).
# Rank - In a multi-node, multi-GPU setup, each process (corresponding to a GPU) is assigned a rank. For example, in an 8-GPU system, ranks might range from 0 to 7.
# The rank helps the training framework manage which part of the model or data a specific process is handling.
# Global Rank: A unique identifier for each process across all nodes.
# Local Rank: The rank within a specific node (for multi-node systems).


def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config["dist_backend"],
            init_method=h.dist_config["dist_url"],
            world_size=h.dist_config["world_size"] * h.num_gpus,
            rank=rank,
        )

    torch.cuda.manual_seed(h.seed)
    device = torch.device("cuda:{:d}".format(rank))

    generator = Generator(h=h).to(device)
    mpd = MultiPeriodDiscriminator(h=h).to(device)
    mrd = MultiResolutionDiscriminator(cfg=h).to(device)

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        # checks for the generator and discriminator weight matrices
        cp_g = scan_checkpoint(a.checkpoint_path, "g_")
        cp_do = scan_checkpoint(a.checkpoint_path, "do_")

    steps = 0

    # we only have the generator weights present, so use them for loading generator
    # keeping the discriminator weights randomly initialized
    if cp_g is not None:
        state_dict_g = load_checkpoint(cp_g, device)
        generator.load_state_dict(state_dict_g["generator"])

    if cp_do is not None:
        state_dict_do = load_checkpoint(cp_do, device)
        mpd.load_state_dict(state_dict_do["mpd"])
        mrd.load_state_dict(state_dict_do["mrd"])
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        mrd = DistributedDataParallel(mrd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mpd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay, last_epoch=last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.lr_decay, last_epoch=last_epoch
    )

    # use dataset_utils.py to create data manifest and acoustic codes
    trainset = UnitDataset(
        data_manifest="data/train.tsv",
        acoustic_data_path="data/train_acoutic_codes.npy",
        codebook_num=h.codebook_num,
        segment_size=h.segment_size,
        unit_hop_size=h.unit_hop_size,
        n_fft=h.n_fft,
        # num_mels at 80 right now, consider increasing them to 128 as we're training on music now not vocals
        # can also consider increasing the
        num_mels=h.num_mels,
        hop_size=h.hop_size,
        win_size=h.win_size,
        sampling_rate=h.sampling_rate,
        fmin=h.fmin,
        fmax=h.fmax,
        split=True,
        shuffle=True,
        n_cache_reuse=1,
        device=None,
        fmax_loss=None,
        fine_tuning=False,
        base_mels_path=None,
        is_seen=True,
    )

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=True,
        sampler=train_sampler,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
    )

    if rank == 0:
        validset = UnitDataset(
            data_manifest="data/valid.tsv",
            acoustic_data_path="data/valid_acoutic_codes.npy",
            codebook_num=h.codebook_num,
            segment_size=h.segment_size,
            unit_hop_size=h.unit_hop_size,
            n_fft=h.n_fft,
            # num_mels at 80 right now, consider increasing them to 128 as we're training on music now not vocals
            # can also consider increasing the
            num_mels=h.num_mels,
            hop_size=h.hop_size,
            win_size=h.win_size,
            sampling_rate=h.sampling_rate,
            fmin=h.fmin,
            fmax=h.fmax,
            split=True,
            shuffle=True,
            n_cache_reuse=1,
            device=None,
            fmax_loss=None,
            fine_tuning=False,
            base_mels_path=None,
            is_seen=True,
        )
        
        validation_loader = DataLoader(
            validset,
            num_workers=1,
            shuffle=True,
            sampler=None,
            batch_size=h.batch_size,
            pin_memory=True,
            drop_last=True,
        )

        sw = SummaryWriter(os.path.join(a.checkpoint_path, "logs"))

    generator.train()
    mpd.train()
    mrd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            # acoustic_tokens, audio, filename, mel_loss
            x, y, _, y_mel = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)
            
            # get the generated audio
            y_g_hat = generator(x)
            # convert the generated audio to mel_spectrogram
            y_g_hat_mel = mel_spectrogram(
                y_g_hat.squeeze(1),
                h.n_fft,
                h.num_mels,
                h.sampling_rate,
                h.hop_size,
                h.win_size,
                h.fmin,
                h.fmax_for_loss,
            )

            optim_d.zero_grad()

            # the real and generated audios are sent through the discriminators
            # we get the discrimnator reprn of real and generated audios, which are used for calculating loss terms

            # MPD
            y_dp_hat_r, y_dp_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_p, losses_disc_p_r, losses_disc_p_g = discriminator_loss(
                y_dp_hat_r, y_dp_hat_g
            )

            # MRD
            y_dr_hat_r, y_dr_hat_g, _, _ = mrd(y, y_g_hat.detach())
            loss_disc_r, losses_disc_r_r, losses_disc_r_g = discriminator_loss(
                y_dr_hat_r, y_dr_hat_g
            )

            loss_disc_all = loss_disc_r + loss_disc_p

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_dp_hat_r, y_dp_hat_g, fmap_p_r, fmap_p_g = mpd(y, y_g_hat)
            y_dr_hat_r, y_dr_hat_g, fmap_r_r, fmap_r_g = mrd(y, y_g_hat)
            loss_fm_p = feature_loss(fmap_p_r, fmap_p_g)
            loss_fm_r = feature_loss(fmap_r_r, fmap_r_g)
            loss_gen_p, losses_gen_p = generator_loss(y_dp_hat_g)
            loss_gen_r, losses_gen_r = generator_loss(y_dr_hat_g)
            loss_gen_all = loss_gen_p + loss_gen_r + loss_fm_p + loss_fm_r + loss_mel

            loss_gen_all.backward()
            optim_g.step()
            
            # look at the lines below, lines above are edited for unit vocoding 
            
            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print(
                        "Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}".format(
                            steps, loss_gen_all, mel_error, time.time() - start_b
                        )
                    )

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "generator": (
                                generator.module if h.num_gpus > 1 else generator
                            ).state_dict()
                        },
                    )
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "mpd": (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                            "mrd": (mrd.module if h.num_gpus > 1 else mrd).state_dict(),
                            "optim_g": optim_g.state_dict(),
                            "optim_d": optim_d.state_dict(),
                            "steps": steps,
                            "epoch": epoch,
                        },
                    )

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel = batch
                            y_g_hat = generator(x.to(device))
                            y_mel = torch.autograd.Variable(
                                y_mel.to(device, non_blocking=True)
                            )
                            y_g_hat_mel = mel_spectrogram(
                                y_g_hat.squeeze(1),
                                h.n_fft,
                                h.num_mels,
                                h.sampling_rate,
                                h.hop_size,
                                h.win_size,
                                h.fmin,
                                h.fmax_for_loss,
                            )
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio(
                                        "gt/y_{}".format(j),
                                        y[0],
                                        steps,
                                        h.sampling_rate,
                                    )
                                    sw.add_figure(
                                        "gt/y_spec_{}".format(j),
                                        plot_spectrogram(x[0]),
                                        steps,
                                    )

                                sw.add_audio(
                                    "generated/y_hat_{}".format(j),
                                    y_g_hat[0],
                                    steps,
                                    h.sampling_rate,
                                )
                                y_hat_spec = mel_spectrogram(
                                    y_g_hat.squeeze(1),
                                    h.n_fft,
                                    h.num_mels,
                                    h.sampling_rate,
                                    h.hop_size,
                                    h.win_size,
                                    h.fmin,
                                    h.fmax,
                                )
                                sw.add_figure(
                                    "generated/y_hat_spec_{}".format(j),
                                    plot_spectrogram(
                                        y_hat_spec.squeeze(0).cpu().numpy()
                                    ),
                                    steps,
                                )

                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print(
                "Time taken for epoch {} is {} sec\n".format(
                    epoch + 1, int(time.time() - start)
                )
            )


def main():
    print("Initializing Training Process..")

    parser = argparse.ArgumentParser()

    parser.add_argument("--group_name", default=None)
    parser.add_argument("--input_wavs_dir", default="LJSpeech-1.1/wavs")
    parser.add_argument("--input_mels_dir", default="ft_dataset")
    parser.add_argument("--input_training_file", default="LJSpeech-1.1/training.txt")
    parser.add_argument(
        "--input_validation_file", default="LJSpeech-1.1/validation.txt"
    )
    parser.add_argument("--checkpoint_path", default="cp_hifigan")
    parser.add_argument("--config", default="")
    parser.add_argument("--training_epochs", default=3100, type=int)
    parser.add_argument("--stdout_interval", default=5, type=int)
    parser.add_argument("--checkpoint_interval", default=5000, type=int)
    parser.add_argument("--summary_interval", default=100, type=int)
    parser.add_argument("--validation_interval", default=1000, type=int)
    parser.add_argument("--fine_tuning", default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, "config.json", a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print("Batch size per GPU :", h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(
            train,
            nprocs=h.num_gpus,
            args=(
                a,
                h,
            ),
        )
    else:
        train(0, a, h)


if __name__ == "__main__":
    main()
