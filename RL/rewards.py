import numpy as np
from rdkit import Chem

from SA_Score import sascorer


def smiles_is_valid(smiles: str) -> bool:
    if not smiles or not smiles.strip():
        return False
    return Chem.MolFromSmiles(smiles) is not None


def calculate_sascore(smiles: str) -> float:
    try:
        if not smiles_is_valid(smiles):
            return 10.0
        return sascorer.calculateScore(Chem.MolFromSmiles(smiles))
    except Exception:
        return 10.0


def compute_combined_reward(
    detonation_velocity: float,
    sascore: float,
    density: float,
    is_valid: bool,
    w_d: float = 1.0,
    w_sa: float = 1.0,
    w_rho: float = 1.0,
    invalid_penalty: float = -5.0,
    enable_bonus: bool = True,
) -> float:
    if not is_valid:
        return invalid_penalty

    d_norm = detonation_velocity / 10.0
    sa_norm = sascore / 10.0
    rho_norm = density / 2.0
    reward = w_d * d_norm - w_sa * sa_norm + w_rho * rho_norm
    if enable_bonus:
        d_bonus = 1.0 if detonation_velocity > 8.5 else 0.0
        rho_bonus = 1.0 if density > 1.8 else 0.0
        reward = reward + d_bonus + rho_bonus
    return reward


def compute_reward_with_baseline(rewards: np.ndarray) -> np.ndarray:
    return rewards - np.mean(rewards)
