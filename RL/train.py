
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*MorganGenerator.*')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from dataclasses import asdict, dataclass
from typing import List, Tuple, Optional
import os
import copy
import time
import argparse
import random
from multiprocessing import Pool
from collections import deque
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


from pathlib import Path
import yaml
from torch.nn.utils.rnn import pad_sequence
from generator.model import GPTGenerator
from generator.tokenizer import SmilesTokenizer
from predictor.inference import PropertyPredictor
from RL.rewards import (
    calculate_sascore, smiles_is_valid, compute_combined_reward,
    compute_reward_with_baseline,
)
from RL.metrics import (
    calculate_validity, calculate_uniqueness, calculate_diversity, MetricsTracker,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batched(iterable, batch_size: int):
    l = len(iterable)
    for i in range(0, l, batch_size):
        yield iterable[i:i+batch_size]


class ExperienceReplay:
    def __init__(self, max_size: int = 1000, pad_idx: int = 0):
        self.max_size = max_size
        self.memory = deque(maxlen=max_size)
        self.pad_idx = pad_idx

    def add(self, seqs: torch.Tensor, scores: np.ndarray, prior_logp: np.ndarray):
        for seq, score, logp in zip(seqs, scores, prior_logp):
            self.memory.append((seq, score, logp))

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self.memory) < n:
            n = len(self.memory)

        indices = np.random.choice(len(self.memory), n, replace=False)
        samples = [self.memory[i] for i in indices]


        max_len = max(s[0].size(0) for s in samples)


        padded_seqs = []
        for seq, score, logp in samples:
            cur_len = seq.size(0)
            pad_len = max_len - cur_len
            if pad_len > 0:

                seq = F.pad(seq, (0, pad_len), value=self.pad_idx)
            padded_seqs.append(seq)

        seqs = torch.stack(padded_seqs, dim=0)
        scores = torch.tensor([s[1] for s in samples], dtype=torch.float32)
        prior_logp = torch.tensor([s[2] for s in samples], dtype=torch.float32)

        return seqs, scores, prior_logp

    def __len__(self):
        return len(self.memory)


def sequence_likelihood(model: nn.Module, seqs: torch.Tensor, device: str,
                       requires_grad: bool = False) -> torch.Tensor:
    seqs = seqs.to(device)
    B, S = seqs.size()
    PAD = 0

    pad_mask = (seqs == PAD)

    if requires_grad:
        was_training = model.training
        model.eval()

        logits = model(seqs, pad_mask)
        logits = logits.to(device)
        seqs = seqs.to(device)
        pad_mask = pad_mask.to(device)

        logprobs = F.log_softmax(logits, dim=-1)
        actions = seqs[:, 1:]
        logprobs_taken = logprobs[:, :-1].gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        valid_mask = ~pad_mask[:, 1:]
        logprobs_taken = logprobs_taken * valid_mask
        logprobs_seq = logprobs_taken.sum(dim=1)
        logprobs_seq = torch.clamp(logprobs_seq, min=-1000, max=0)

        if was_training:
            model.train()

        return logprobs_seq
    else:
        was_training = model.training
        model.eval()

        with torch.no_grad():
            logits = model(seqs, pad_mask)
            logits = logits.to(device)
            seqs = seqs.to(device)
            pad_mask = pad_mask.to(device)

            logprobs = F.log_softmax(logits, dim=-1)
            actions = seqs[:, 1:]
            logprobs_taken = logprobs[:, :-1].gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            valid_mask = ~pad_mask[:, 1:]
            logprobs_taken = logprobs_taken * valid_mask
            logprobs_seq = logprobs_taken.sum(dim=1)
            logprobs_seq = torch.clamp(logprobs_seq, min=-1000, max=0)

        if was_training:
            model.train()

        return logprobs_seq


@dataclass
class RLConfig:


    max_len: int = 256
    temperature: float = 1.0
    top_k: int = 30
    gen_batch_size: int = 512


    batch_size: int = 64
    lr: float = 1e-6
    sigma: float = 60.0
    grad_clip: float = 1.0


    use_experience_replay: bool = True
    replay_buffer_size: int = 1000
    replay_batch_size: int = 4


    w_detonation: float = 1.0
    w_sascore: float = 1.0
    w_density: float = 1.0
    invalid_penalty: float = -10.0

    enable_bonus: bool = True


    update_prior_every: int = 500


    steps: int = 5000
    eval_every: int = 500
    eval_n_samples: int = 1000
    eval_batch: int = 512
    save_every: int = 500


    n_workers: int = 4

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./rl_results"


class REINFORCETrainer:
    def __init__(self, gen_model, reg_model_d, reg_model_rho, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.agent = gen_model.to(self.device)
        self.prior = copy.deepcopy(self.agent).requires_grad_(False).eval()
        self.reg_d = reg_model_d.to(self.device).requires_grad_(False).eval()
        self.reg_rho = reg_model_rho.to(self.device).requires_grad_(False).eval()
        self.gen_single = self.agent
        self.tokenizer = SmilesTokenizer(gen_model.vocab)
        self.pad_idx = self.tokenizer.pad_id
        self.start_id = self.tokenizer.start_id
        self.end_id = self.tokenizer.end_id
        self.vocab_size = len(self.tokenizer.vocab)
        self.opt = torch.optim.Adam(self.agent.parameters(), lr=cfg.lr)
        self.experience = (ExperienceReplay(cfg.replay_buffer_size, self.pad_idx)
                           if cfg.use_experience_replay else None)
        self.metrics_tracker = MetricsTracker(save_dir=cfg.save_dir)
        self.step_history = []
        self.reward_history = []
        self.best_reward = -float('inf')


    def _update_prior(self):
        self.prior.load_state_dict(self.agent.state_dict())
        self.prior.requires_grad_(False).eval()

    @torch.no_grad()
    def _sample_from_policy(self, n: int, return_seqs: bool = False) -> Tuple[Optional[torch.Tensor], List[str]]:
        device = self.device
        S = self.cfg.max_len
        temperature = self.cfg.temperature
        top_k = self.cfg.top_k

        model = self.gen_single
        was_training = model.training
        model.eval()

        gen_batch_size = min(self.cfg.gen_batch_size, n)
        all_seqs = [] if return_seqs else None
        all_smiles = []

        pbar = tqdm(total=n, desc="Generating molecules", unit="mol",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                leave=False)

        for start_idx in range(0, n, gen_batch_size):
            batch_n = min(gen_batch_size, n - start_idx)


            generated = torch.full((batch_n, 1), self.start_id, dtype=torch.long, device=device)
            finished = torch.zeros(batch_n, dtype=torch.bool, device=device)

            for step in range(S - 1):
                logits = model(generated)
                logits = logits.detach()
                next_token_logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    topk_logits, topk_indices = torch.topk(next_token_logits, top_k)
                    probs = torch.softmax(topk_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, 1)
                    next_token = torch.gather(topk_indices, dim=1, index=next_token_idx)
                else:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)

                finished = finished | (next_token.squeeze() == self.end_id)

                end_token_tensor = torch.full_like(next_token, self.end_id)
                next_token = torch.where(finished.unsqueeze(1), end_token_tensor, next_token)

                generated = torch.cat([generated, next_token], dim=1)

                if finished.all():
                    break

                del logits, next_token_logits, probs, next_token


            generated_cpu = generated.cpu().numpy()
            finished_cpu = finished.cpu().numpy()

            del generated, finished

            for i in range(batch_n):
                token_ids = generated_cpu[i]

                if finished_cpu[i] and self.end_id in token_ids:
                    end_pos = np.where(token_ids == self.end_id)[0][0]
                    token_ids = token_ids[:end_pos + 1]

                smiles = self.tokenizer.decode(token_ids)

                all_smiles.append(smiles)

            if return_seqs:
                all_seqs.append(torch.from_numpy(generated_cpu))

            del generated_cpu, finished_cpu

            pbar.update(batch_n)


            if device.startswith('cuda'):
                torch.cuda.empty_cache()

        pbar.close()
        del pbar

        if return_seqs:
            all_seqs_tensor = pad_sequence([row for batch in all_seqs for row in batch], batch_first=True, padding_value=self.pad_idx)
            del all_seqs
        else:
            all_seqs_tensor = None

        if was_training:
            model.train()

        if device.startswith('cuda'):
            torch.cuda.empty_cache()


        gc.collect()

        return all_seqs_tensor, all_smiles

    @torch.no_grad()
    def _predict_properties(self, smiles):
        d_predictions = torch.zeros(len(smiles))
        rho_predictions = torch.zeros(len(smiles))
        valid = [i for i, smi in enumerate(smiles) if smiles_is_valid(smi)]
        for rows in batched(valid, self.cfg.eval_batch):
            batch = [smiles[i] for i in rows]
            d_predictions[rows] = self.reg_d(batch).cpu()
            rho_predictions[rows] = self.reg_rho(batch).cpu()
        return d_predictions, rho_predictions

    def _calculate_rewards(self, smiles: List[str], d_predictions: torch.Tensor,
                          rho_predictions: torch.Tensor) -> np.ndarray:
        n = len(smiles)
        rewards = np.zeros(n, dtype=np.float32)


        try:
            if self.cfg.n_workers > 1:
                with Pool(self.cfg.n_workers) as pool:
                    sa_scores = pool.map(calculate_sascore, smiles)
            else:
                sa_scores = [calculate_sascore(s) for s in smiles]
        except Exception as e:
            print(f"Warning: Parallel processing failed ({e})")
            sa_scores = [calculate_sascore(s) for s in smiles]


        for i in range(n):
            is_valid = smiles_is_valid(smiles[i])
            d_val = d_predictions[i].item()
            rho_val = rho_predictions[i].item()
            sa_val = sa_scores[i]

            reward = compute_combined_reward(
                detonation_velocity=d_val,
                sascore=sa_val,
                density=rho_val,
                is_valid=is_valid,
                w_d=self.cfg.w_detonation,
                w_sa=self.cfg.w_sascore,
                w_rho=self.cfg.w_density,
                invalid_penalty=self.cfg.invalid_penalty,
                enable_bonus=self.cfg.enable_bonus
            )
            rewards[i] = reward


        return rewards

    def save_checkpoint(self, step: int, is_best: bool = False):
        save_dir = self.metrics_tracker.save_dir
        os.makedirs(save_dir, exist_ok=True)

        model = self.agent

        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'config': asdict(self.cfg),
            'hyper_parameters': dict(model.hparams),
            'vocab': model.vocab,
            'step_history': self.step_history,
            'reward_history': self.reward_history,
            'best_reward': self.best_reward,
        }

        save_path = os.path.join(save_dir, f'checkpoint_step_{step}.pt')
        torch.save(checkpoint, save_path)
        print(f"[Checkpoint] Saved to {save_path}")

        if is_best:
            best_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"[Best Model] Saved to {best_path} (reward: {self.best_reward:.4f})")

    def train(self):
        cfg = self.cfg
        self.agent.train()


        print("\n" + "-"*70)
        print("Validating Prior-Agent Consistency")
        print("-"*70)
        test_seqs, test_smiles = self._sample_from_policy(10, return_seqs=True)
        with torch.no_grad():
            test_agent_logp = sequence_likelihood(self.agent, test_seqs, self.device, requires_grad=False)
            test_prior_logp = sequence_likelihood(self.prior, test_seqs, self.device, requires_grad=False)
            test_diff = (test_agent_logp - test_prior_logp).abs().mean()
            print(f"Initial likelihood difference: {test_diff:.6f}")
            if test_diff > 1.0:
                print(f"WARNING: Large initial difference!")
            else:
                print(f"Prior and Agent properly initialized")

        del test_seqs, test_smiles, test_agent_logp, test_prior_logp
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()

        print("-"*70)

        device = self.device


        print("\n" + "-"*70)
        print("Initial Evaluation (Step 0)")
        print("-"*70)
        self.evaluate(step=0, n_samples=cfg.eval_n_samples, batch=cfg.eval_batch)

        start_time = time.time()
        ema_reward = None

        for step in range(1, cfg.steps + 1):
            step_start = time.time()


            print(f"\n{'-'*70}")
            print(f"Step {step}/{cfg.steps}: Generating {cfg.batch_size} molecules")
            print("-"*70)

            gen_start = time.time()
            # 采样后计算奖励，再更新策略。
            seqs, smiles = self._sample_from_policy(cfg.batch_size, return_seqs=True)
            gen_time = time.time() - gen_start


            unique_seqs = []
            unique_smiles = []
            seen = set()
            for seq, smi in zip(seqs, smiles):
                seq_np = seq.cpu().numpy() if seq.is_cuda else seq.numpy()
                seq_tuple = tuple(seq_np)
                if seq_tuple not in seen:
                    seen.add(seq_tuple)
                    unique_seqs.append(seq)
                    unique_smiles.append(smi)

            del seqs

            seqs = torch.stack(unique_seqs).to(device)
            smiles = unique_smiles

            del unique_seqs, unique_smiles

            valid_count = sum(1 for s in smiles if smiles_is_valid(s))
            validity = valid_count / len(smiles) * 100

            print(f"[Generation] Completed in {gen_time:.2f}s")
            print(f"[Generation] Unique: {len(smiles)}/{cfg.batch_size}")
            print(f"[Generation] Validity: {validity:.2f}% ({valid_count}/{len(smiles)})")

            print(f"[Generation] Sample SMILES:")
            for i, s in enumerate(smiles[:3]):
                is_valid = "valid" if smiles_is_valid(s) else "invalid"
                print(f"  {i+1}. {s[:50]}... {is_valid}")


            print(f"[Rewards] Computing...")
            reward_start = time.time()

            d_predictions, rho_predictions = self._predict_properties(smiles)
            rewards = self._calculate_rewards(smiles, d_predictions, rho_predictions)

            rewards_baseline = compute_reward_with_baseline(rewards)

            reward_time = time.time() - reward_start
            print(f"[Rewards] Completed in {reward_time:.2f}s")
            print(f"[Rewards] Raw Mean: {np.mean(rewards):.4f}, Adjusted: {np.mean(rewards_baseline):.4f}")

            del d_predictions, rho_predictions


            agent_logp = sequence_likelihood(self.agent, seqs, device, requires_grad=True)
            with torch.no_grad():
                prior_logp = sequence_likelihood(self.prior, seqs, device, requires_grad=False)

            rewards_tensor = torch.tensor(rewards_baseline, dtype=torch.float32, device=device)
            augmented_logp = prior_logp + cfg.sigma * rewards_tensor


            loss = torch.pow(augmented_logp - agent_logp, 2)


            if cfg.use_experience_replay and len(self.experience) >= cfg.replay_batch_size:
                exp_seqs, exp_rewards, exp_prior_logp = self.experience.sample(cfg.replay_batch_size)
                exp_seqs = exp_seqs.to(device)
                exp_rewards = exp_rewards.to(device)
                exp_prior_logp = exp_prior_logp.to(device)

                exp_agent_logp = sequence_likelihood(self.agent, exp_seqs, device, requires_grad=True)
                exp_augmented_logp = exp_prior_logp + cfg.sigma * exp_rewards
                exp_loss = torch.pow(exp_augmented_logp - exp_agent_logp, 2)

                loss = torch.cat((loss, exp_loss), 0)
                agent_logp = torch.cat((agent_logp, exp_agent_logp), 0)

                del exp_seqs, exp_rewards, exp_prior_logp, exp_agent_logp, exp_augmented_logp, exp_loss

            if cfg.use_experience_replay:
                self.experience.add(seqs.cpu(), rewards, prior_logp.cpu().numpy())

            loss = loss.mean()
            loss_reg = -(1 / agent_logp.abs().clamp(min=1e-6)).mean()
            loss = loss + 5 * 1e3 * loss_reg

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] Loss is nan/inf at step {step}, skipping")
                del seqs, rewards, rewards_baseline, rewards_tensor
                del agent_logp, prior_logp, augmented_logp, loss, loss_reg
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                continue


            train_start = time.time()
            self.opt.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.grad_clip)

            self.opt.step()
            train_time = time.time() - train_start


            if step % cfg.update_prior_every == 0:
                self._update_prior()


            with torch.no_grad():
                batch_mean_r = np.mean(rewards)
                ema_reward = batch_mean_r if ema_reward is None else 0.9 * ema_reward + 0.1 * batch_mean_r

                agent_logp_mean = agent_logp.mean().item()
                prior_logp_mean = prior_logp.mean().item()
                augmented_logp_mean = augmented_logp.mean().item()

                loss_value = loss.item()
                loss_reg_value = loss_reg.item()

            self.step_history.append(step)
            self.reward_history.append(ema_reward)

            step_time = time.time() - step_start
            print(f"\n[Step {step}] Summary:")
            print(f"  Loss:       {loss_value:.4f} (Reg: {loss_reg_value:.4f})")
            print(f"  Reward:     raw_mean={batch_mean_r:.4f}, ema={ema_reward:.4f}")
            print(f"  Likelihood: agent={agent_logp_mean:.2f}, prior={prior_logp_mean:.2f}, aug={augmented_logp_mean:.2f}")
            print(f"  Validity:   {validity:.2f}% ({valid_count}/{len(smiles)})")
            if cfg.use_experience_replay:
                print(f"  Replay:     {len(self.experience)}/{cfg.replay_buffer_size} experiences")
            print(f"  Time:       gen={gen_time:.1f}s, reward={reward_time:.1f}s, train={train_time:.1f}s, total={step_time:.1f}s")

            print(f"\n[Top 5 Molecules]")
            top_indices = np.argsort(rewards)[-5:][::-1]
            for idx in top_indices:
                print(f"  Score: {rewards[idx]:.4f} | {smiles[idx][:60]}")


            del seqs, rewards, rewards_baseline, rewards_tensor
            del agent_logp, prior_logp, augmented_logp, loss, loss_reg

            if device.startswith('cuda'):
                torch.cuda.empty_cache()


            if step % 10 == 0:
                gc.collect()


            if step % cfg.save_every == 0 or step == cfg.steps:
                is_best = ema_reward > self.best_reward
                if is_best:
                    self.best_reward = ema_reward
                self.save_checkpoint(step, is_best=is_best)


            if step % cfg.eval_every == 0 or step == cfg.steps:
                print(f"\n{'-'*70}")
                print(f"Evaluation at Step {step}")
                print("-"*70)
                self.evaluate(step=step, n_samples=cfg.eval_n_samples, batch=cfg.eval_batch)

                self.metrics_tracker.plot_training_curves(self.step_history, self.reward_history)
                self.metrics_tracker.plot_property_distributions()
                self.metrics_tracker.plot_metrics_evolution()

                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                gc.collect()

        total_time = time.time() - start_time
        print(f"\n{'-'*70}")
        print(f"Training Completed!")
        print(f"Total time:   {total_time:.2f}s ({total_time/60:.2f}min)")
        print(f"Best reward:  {self.best_reward:.4f}")
        print(f"Final reward: {ema_reward:.4f}")
        print("-"*70)

        self.save_checkpoint(cfg.steps, is_best=False)
        self.metrics_tracker.plot_training_curves(self.step_history, self.reward_history)
        self.metrics_tracker.plot_property_distributions()
        self.metrics_tracker.plot_metrics_evolution()
        self.metrics_tracker.generate_summary_report()

    @torch.no_grad()
    def evaluate(self, step: int, n_samples: int, batch: int = 512):
        cfg = self.cfg
        self.agent.eval()

        print(f"\n[Evaluation] Generating exactly {n_samples} molecules...")

        all_smiles = []

        pbar = tqdm(total=n_samples, desc=f"Step {step} - Generating", unit="mol", leave=False)

        for start_idx in range(0, n_samples, batch):
            cur_batch = min(batch, n_samples - start_idx)

            try:
                _, smiles = self._sample_from_policy(cur_batch, return_seqs=False)
                all_smiles.extend(smiles)
                pbar.update(cur_batch)
                del smiles

            except Exception as e:
                print(f"\n Error during generation: {e}")
                if self.device.startswith('cuda'):
                    torch.cuda.empty_cache()

                cur_batch = max(1, cur_batch // 2)
                try:
                    _, smiles = self._sample_from_policy(cur_batch, return_seqs=False)
                    all_smiles.extend(smiles)
                    pbar.update(cur_batch)
                    del smiles
                except Exception as e2:
                    print(f"Failed again: {e2}")
                    break

            if (start_idx // batch) % 2 == 0 and self.device.startswith('cuda'):
                torch.cuda.empty_cache()

        pbar.close()
        del pbar

        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()


        valid_indices = []
        invalid_indices = []
        for i, s in enumerate(all_smiles):
            if smiles_is_valid(s):
                valid_indices.append(i)
            else:
                invalid_indices.append(i)

        n_valid = len(valid_indices)
        n_invalid = len(invalid_indices)
        n_total = len(all_smiles)

        print(f"\n[Generation Summary]")
        print(f"  Total generated: {n_total}")
        print(f"  Valid:           {n_valid} ({100.0*n_valid/n_total:.2f}%)")
        print(f"  Invalid:         {n_invalid} ({100.0*n_invalid/n_total:.2f}%)")

        valid_smiles = [all_smiles[i] for i in valid_indices]

        if n_valid == 0:
            print("\n WARNING: No valid molecules generated!")
            self.agent.train()
            return

        print(f"\n[Properties] Computing for {n_valid} valid molecules...")

        d_preds, rho_preds = self._predict_properties(valid_smiles)

        try:
            if cfg.n_workers > 1:
                with Pool(cfg.n_workers) as pool:
                    sa_scores = pool.map(calculate_sascore, valid_smiles)
            else:
                sa_scores = [calculate_sascore(s) for s in valid_smiles]
        except Exception as e:
            print(f"\n Warning: Parallel SAscore failed ({e})")
            sa_scores = [calculate_sascore(s) for s in valid_smiles]

        all_d_values = [d_preds[i].item() for i in range(n_valid)]
        all_rho_values = [rho_preds[i].item() for i in range(n_valid)]
        all_sa_values = sa_scores
        all_rewards = []

        for i in range(n_valid):
            reward = compute_combined_reward(
                detonation_velocity=all_d_values[i],
                sascore=all_sa_values[i],
                density=all_rho_values[i],
                is_valid=True,
                w_d=cfg.w_detonation,
                w_sa=cfg.w_sascore,
                w_rho=cfg.w_density,
                enable_bonus=cfg.enable_bonus
            )
            all_rewards.append(reward)

        del d_preds, rho_preds

        validity = calculate_validity(all_smiles)
        uniqueness = calculate_uniqueness(all_smiles)
        diversity = calculate_diversity(valid_smiles)

        print(f"\n{'-'*70}")
        print(f"Evaluation Results - Step {step}")
        print(f"{'-'*70}")
        print(f"Sample Statistics:")
        print(f"  Total generated: {n_total}")
        print(f"  Valid molecules: {n_valid}")
        print(f"  Invalid molecules: {n_invalid}")
        print(f"\nQuality Metrics:")
        print(f"  Validity:        {validity:.2f}%")
        print(f"  Uniqueness:      {uniqueness:.2f}%")
        print(f"  Diversity:       {diversity:.4f}")

        if n_valid > 0:
            print(f"\nProperty Statistics (Valid Molecules Only):")
            print(f"  D: {np.mean(all_d_values):.3f} ± {np.std(all_d_values):.3f}")
            print(f"  Density:             {np.mean(all_rho_values):.3f} ± {np.std(all_rho_values):.3f} g/cm³")
            print(f"  SA Score:            {np.mean(all_sa_values):.3f} ± {np.std(all_sa_values):.3f}")
            print(f"  Combined Reward:     {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
        print(f"{'-'*70}\n")

        epoch = step // cfg.eval_every if step > 0 else 0

        if n_valid > 0:
            self.metrics_tracker.add_eval_data(
                epoch=epoch,
                d_values=all_d_values,
                rho_values=all_rho_values,
                sa_values=all_sa_values,
                rewards=all_rewards,
                step=step
            )

        self.metrics_tracker.record_metrics(
            epoch=epoch,
            validity=validity,
            uniqueness=uniqueness,
            diversity=diversity,
            n_attempts=n_total,
            n_valid=n_valid,
            step=step
        )

        if n_valid > 0:
            n_unique = self.metrics_tracker.record_molecules(
                epoch=epoch,
                smiles_list=valid_smiles,
                d_values=all_d_values,
                rho_values=all_rho_values,
                sa_values=all_sa_values,
                rewards=all_rewards
            )

            self.metrics_tracker.record_properties(epoch, all_d_values, all_rho_values, all_sa_values)

            print(f"[Summary] Saved {n_unique} unique valid molecules")

        if epoch > 0:
            self.metrics_tracker.plot_metrics_evolution()
            self.metrics_tracker.plot_property_distributions()

        print(f"[Summary] Evaluated {n_total} molecules ({n_valid} valid, {n_invalid} invalid)")

        del all_smiles, valid_smiles, all_d_values, all_rho_values, all_sa_values, all_rewards
        del valid_indices, invalid_indices

        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()

        gc.collect()

        self.agent.train()


def load_models(generator_checkpoint, d_config, rho_config, device):
    state = torch.load(generator_checkpoint, map_location='cpu', weights_only=False)
    generator = GPTGenerator(**state['hyper_parameters'])
    generator.load_state_dict(state.get('state_dict', state.get('model_state_dict')))
    reg_d = PropertyPredictor(**d_config)
    reg_rho = PropertyPredictor(**rho_config)
    return generator.to(device), reg_d.to(device), reg_rho.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='RL/configs/base.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    train_cfg = RLConfig(**cfg['training'])
    set_seed(train_cfg.seed)
    generator, reg_d, reg_rho = load_models(
        cfg['generator_checkpoint'], cfg['predictor_d'], cfg['predictor_rho'], train_cfg.device)
    trainer = REINFORCETrainer(generator, reg_d, reg_rho, train_cfg)
    trainer.train()


if __name__ == '__main__':
    main()
