import csv
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy import stats

from RL.rewards import smiles_is_valid


def calculate_validity(smiles_list: List[str]) -> float:
    """Calculate validity percentage."""
    if not smiles_list:
        return 0.0
    valid_count = sum(1 for s in smiles_list if smiles_is_valid(s))
    return 100.0 * valid_count / len(smiles_list)


def calculate_uniqueness(smiles_list: List[str]) -> float:
    """Calculate uniqueness percentage (canonical SMILES)."""
    if not smiles_list:
        return 0.0
    
    canonical_smiles = []
    for s in smiles_list:
        if smiles_is_valid(s):
            mol = Chem.MolFromSmiles(s)
            canonical = Chem.MolToSmiles(mol, canonical=True)
            canonical_smiles.append(canonical)
    
    if not canonical_smiles:
        return 0.0
    
    unique_count = len(set(canonical_smiles))
    return 100.0 * unique_count / len(canonical_smiles)


def calculate_diversity(smiles_list: List[str], radius: int = 2, nBits: int = 2048) -> float:
    """Calculate diversity using Tanimoto distance (average pairwise)."""
    if len(smiles_list) < 2:
        return 0.0
    
    # Get fingerprints
    fps = []
    for s in smiles_list:
        if smiles_is_valid(s):
            mol = Chem.MolFromSmiles(s)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            fps.append(fp)
    
    if len(fps) < 2:
        return 0.0
    
    # Calculate pairwise distances
    total_distance = 0.0
    count = 0
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            total_distance += (1 - similarity)
            count += 1
    
    return total_distance / count if count > 0 else 0.0


class MetricsTracker:
    """Track and save training metrics to CSV files."""
    
    def __init__(self, save_dir: str = "./results"):
        self.save_dir = save_dir
        self.molecules_dir = os.path.join(save_dir, "molecules")
        self.metrics_dir = os.path.join(save_dir, "metrics")
        self.plots_dir = os.path.join(save_dir, "plots")
        self.data_dir = os.path.join(save_dir, "plot_data")  
        
        # Create directories
        os.makedirs(self.molecules_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize metrics CSV
        self.metrics_csv_path = os.path.join(self.metrics_dir, "training_metrics.csv")
        self._init_metrics_csv()
        
        # Store evaluation data for plotting
        self.eval_data = {
            'epochs': [],
            'steps': [],
            'detonation_velocities': [],
            'densities': [],  
            'sascores': [],
            'rewards': []
        }
    
    def _init_metrics_csv(self):
        """Initialize metrics CSV file with headers (added density columns)."""
        if not os.path.exists(self.metrics_csv_path):
            with open(self.metrics_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch',
                    'step',
                    'validity (%)',
                    'uniqueness (%)',
                    'diversity',
                    'n_attempts',
                    'n_valid',
                    'mean_detonation_velocity',
                    'std_detonation_velocity',
                    'mean_density',  
                    'std_density',    
                    'mean_sascore',
                    'std_sascore',
                    'mean_reward',
                    'std_reward'
                ])
    
    def record_properties(self, epoch: int, d_values: List[float], rho_values: List[float], sa_values: List[float]):
        """
        Save molecular properties to CSV (added density).
        
        Creates: molecules/epoch_{epoch}_properties.csv
        """
        csv_path = os.path.join(self.molecules_dir, f"epoch_{epoch}_properties.csv")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['molecule_id', 'detonation_velocity', 'density', 'sascore'])
            
            for i, (d, rho, sa) in enumerate(zip(d_values, rho_values, sa_values)):
                writer.writerow([i, d, rho, sa])
        
        print(f"[CSV] Saved properties to {csv_path}")
    
    def record_molecules(
        self,
        epoch: int,
        smiles_list: List[str],
        d_values: List[float],
        rho_values: List[float],  
        sa_values: List[float],
        rewards: List[float]
    ):
        """
        Save unique valid molecules to CSV (added density).
        
        Creates: molecules/epoch_{epoch}_molecules.csv
        """
        # Get unique valid molecules
        unique_data = {}
        for s, d, rho, sa, r in zip(smiles_list, d_values, rho_values, sa_values, rewards):
            if smiles_is_valid(s):
                mol = Chem.MolFromSmiles(s)
                canonical = Chem.MolToSmiles(mol, canonical=True)
                
                # Keep first occurrence
                if canonical not in unique_data:
                    unique_data[canonical] = {
                        'detonation_velocity': d,
                        'density': rho,
                        'sascore': sa,
                        'reward': r
                    }
        
        # Save to CSV
        csv_path = os.path.join(self.molecules_dir, f"epoch_{epoch}_molecules.csv")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['smiles', 'detonation_velocity', 'density', 'sascore', 'reward'])
            
            for smiles, props in unique_data.items():
                writer.writerow([
                    smiles,
                    props['detonation_velocity'],
                    props['density'],
                    props['sascore'],
                    props['reward']
                ])
        
        print(f"[CSV] Saved {len(unique_data)} unique molecules to {csv_path}")
        return len(unique_data)
    
    def add_eval_data(
        self,
        epoch: int,
        d_values: List[float],
        rho_values: List[float],  
        sa_values: List[float],
        rewards: List[float],
        step: Optional[int] = None
    ):
        """Store evaluation data for plotting and metrics."""
        self.eval_data['epochs'].append(epoch)
        self.eval_data['steps'].append(step if step is not None else epoch)
        self.eval_data['detonation_velocities'].append(d_values)
        self.eval_data['densities'].append(rho_values)  
        self.eval_data['sascores'].append(sa_values)
        self.eval_data['rewards'].append(rewards)
    
    def record_metrics(
        self,
        epoch: int,
        validity: float,
        uniqueness: float,
        diversity: float,
        n_attempts: int,
        n_valid: int,
        step: Optional[int] = None
    ):
        """
        Append metrics to training_metrics.csv (added density statistics).
        
        Args:
            epoch: Evaluation epoch
            validity: Validity percentage
            uniqueness: Uniqueness percentage
            diversity: Diversity score
            n_attempts: Total molecules generated
            n_valid: Number of valid molecules
            step: Training step (optional)
        """
        # Get property statistics from stored eval_data
        mean_d = std_d = mean_rho = std_rho = mean_sa = std_sa = mean_r = std_r = 0.0
        
        if epoch in self.eval_data['epochs']:
            idx = self.eval_data['epochs'].index(epoch)
            d_values = self.eval_data['detonation_velocities'][idx]
            rho_values = self.eval_data['densities'][idx]
            sa_values = self.eval_data['sascores'][idx]
            rewards = self.eval_data['rewards'][idx]
            
            if len(d_values) > 0:
                mean_d = np.mean(d_values)
                std_d = np.std(d_values)
            if len(rho_values) > 0:
                mean_rho = np.mean(rho_values)
                std_rho = np.std(rho_values)
            if len(sa_values) > 0:
                mean_sa = np.mean(sa_values)
                std_sa = np.std(sa_values)
            if len(rewards) > 0:
                mean_r = np.mean(rewards)
                std_r = np.std(rewards)
        
        # Append to CSV
        with open(self.metrics_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                step if step is not None else epoch * 1000,
                validity,
                uniqueness,
                diversity,
                n_attempts,
                n_valid,
                mean_d,
                std_d,
                mean_rho,  
                std_rho,   
                mean_sa,
                std_sa,
                mean_r,
                std_r
            ])
    
    def plot_training_curves(self, step_history: List[int], reward_history: List[float]):
        """Plot training reward curve and save data."""
        if not step_history or not reward_history:
            return
        
        # Save raw data for custom plotting
        data_path = os.path.join(self.data_dir, 'training_curve_data.csv')
        df = pd.DataFrame({
            'step': step_history,
            'ema_reward': reward_history
        })
        df.to_csv(data_path, index=False)
        print(f"[Data] Saved training curve data to {data_path}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(step_history, reward_history, linewidth=2, color='blue', alpha=0.8)
        ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('EMA Reward', fontsize=12, fontweight='bold')
        #ax.set_title('Training Progress: Reward vs Step', fontsize=14, fontweight='bold')
        ax.tick_params(labelsize=11)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'training_reward_curve.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Plot] Saved training curve to {save_path}")
    
    def plot_property_distributions(self):
        """Plot KDE distributions for detonation velocity, density, and SA score across epochs.
        
        MODIFIED: Reads data from epoch_*_molecules.csv files instead of in-memory eval_data.
        This ensures KDE plots show only unique valid molecules saved during evaluation.
        """
        # Find all molecules CSV files
        molecules_files = []
        for fname in os.listdir(self.molecules_dir):
            if fname.startswith('epoch_') and fname.endswith('_molecules.csv'):
                try:
                    epoch_num = int(fname.split('_')[1])
                    molecules_files.append((epoch_num, fname))
                except (ValueError, IndexError):
                    continue
        
        if not molecules_files:
            print("[Warning] No molecules CSV files found, skipping KDE plot")
            return
        
        # Sort by epoch
        molecules_files.sort(key=lambda x: x[0])
        epochs = [ep for ep, _ in molecules_files]
        n_epochs = len(epochs)
        
        print(f"[KDE Plot] Found {n_epochs} epochs with molecule data: {epochs}")
        
        # Load data from CSV files
        epoch_data = {}
        plot_data = {
            'epoch': [],
            'property': [],
            'values': []
        }
        
        for epoch, fname in molecules_files:
            csv_path = os.path.join(self.molecules_dir, fname)
            try:
                df = pd.read_csv(csv_path)
                
                # Extract properties
                d_values = df['detonation_velocity'].values.tolist()
                rho_values = df['density'].values.tolist()
                sa_values = df['sascore'].values.tolist()
                
                epoch_data[epoch] = {
                    'detonation_velocity': d_values,
                    'density': rho_values,
                    'sascore': sa_values
                }
                
                # Accumulate for raw data export
                plot_data['epoch'].extend([epoch] * len(d_values))
                plot_data['property'].extend(['detonation_velocity'] * len(d_values))
                plot_data['values'].extend(d_values)
                
                plot_data['epoch'].extend([epoch] * len(rho_values))
                plot_data['property'].extend(['density'] * len(rho_values))
                plot_data['values'].extend(rho_values)
                
                plot_data['epoch'].extend([epoch] * len(sa_values))
                plot_data['property'].extend(['sascore'] * len(sa_values))
                plot_data['values'].extend(sa_values)
                
                print(f"  Loaded epoch {epoch}: {len(d_values)} molecules")
            except Exception as e:
                print(f"[Warning] Failed to load {fname}: {e}")
                continue
        
        if not epoch_data:
            print("[Warning] No valid data loaded from molecules CSV files")
            return
        
        # Save raw data for custom plotting
        df_plot = pd.DataFrame(plot_data)
        data_path = os.path.join(self.data_dir, 'kde_distribution_data.csv')
        df_plot.to_csv(data_path, index=False)
        print(f"[Data] Saved KDE distribution data to {data_path}")
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_epochs, 2)))
        
        # Plot 1: Detonation Velocity KDE
        ax1 = axes[0]
        for i, epoch in enumerate(epochs):
            d_values = epoch_data[epoch]['detonation_velocity']
            
            if len(d_values) < 2:
                continue
            
            if epoch == 0:
                label = 'Untrained'
                color = 'gray'
                alpha = 0.6
                linewidth = 2.5
            else:
                label = f'Epoch-{epoch}'
                color = colors[i % len(colors)]
                alpha = 0.5
                linewidth = 2
            
            try:
                kde = stats.gaussian_kde(d_values)
                x_min, x_max = min(d_values), max(d_values)
                x_range = np.linspace(x_min, x_max, 300)
                density = kde(x_range)
                
                ax1.plot(x_range, density, label=label, color=color, 
                        alpha=alpha+0.2, linewidth=linewidth)
                ax1.fill_between(x_range, density, alpha=alpha*0.4, color=color)
            except Exception as e:
                print(f"Warning: KDE failed for epoch {epoch} (D): {e}")
        
        ax1.set_xlabel('D (km/s)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Density', fontsize=13, fontweight='bold')
        ax1.set_title('Detonation Velocity Distribution', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10, framealpha=0.9, prop={'weight': 'bold'})
        ax1.tick_params(labelsize=11)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontweight('bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(left=0)
        
        # Plot 2: Density KDE
        ax2 = axes[1]
        for i, epoch in enumerate(epochs):
            rho_values = epoch_data[epoch]['density']
            
            if len(rho_values) < 2:
                continue
            
            if epoch == 0:
                label = 'Untrained'
                color = 'gray'
                alpha = 0.6
                linewidth = 2.5
            else:
                label = f'Epoch-{epoch}'
                color = colors[i % len(colors)]
                alpha = 0.5
                linewidth = 2
            
            try:
                kde = stats.gaussian_kde(rho_values)
                x_min, x_max = max(0, min(rho_values)), min(3, max(rho_values))
                x_range = np.linspace(x_min, x_max, 300)
                density = kde(x_range)
                
                ax2.plot(x_range, density, label=label, color=color,
                        alpha=alpha+0.2, linewidth=linewidth)
                ax2.fill_between(x_range, density, alpha=alpha*0.4, color=color)
            except Exception as e:
                print(f"Warning: KDE failed for epoch {epoch} (Density): {e}")
        
        ax2.set_xlabel('Density (g/cm³)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Density', fontsize=13, fontweight='bold')
        ax2.set_title('Molecular Density Distribution', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10, framealpha=0.9, prop={'weight': 'bold'})
        ax2.tick_params(labelsize=11)
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontweight('bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim([0, 3])
        
        # Plot 3: SA Score KDE
        ax3 = axes[2]
        for i, epoch in enumerate(epochs):
            sa_values = epoch_data[epoch]['sascore']
            
            if len(sa_values) < 2:
                continue
            
            if epoch == 0:
                label = 'Untrained'
                color = 'gray'
                alpha = 0.6
                linewidth = 2.5
            else:
                label = f'Epoch-{epoch}'
                color = colors[i % len(colors)]
                alpha = 0.5
                linewidth = 2
            
            try:
                kde = stats.gaussian_kde(sa_values)
                x_min, x_max = max(1, min(sa_values)), min(10, max(sa_values))
                x_range = np.linspace(x_min, x_max, 300)
                density = kde(x_range)
                
                ax3.plot(x_range, density, label=label, color=color,
                        alpha=alpha+0.2, linewidth=linewidth)
                ax3.fill_between(x_range, density, alpha=alpha*0.4, color=color)
            except Exception as e:
                print(f"Warning: KDE failed for epoch {epoch} (SA): {e}")
        
        ax3.set_xlabel('SAscore', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Density', fontsize=13, fontweight='bold')
        ax3.set_title('Synthetic Accessibility Distribution', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10, framealpha=0.9, prop={'weight': 'bold'})
        ax3.tick_params(labelsize=11)
        for label in ax3.get_xticklabels() + ax3.get_yticklabels():
            label.set_fontweight('bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xlim([0, 10])
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'kde_distributions.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"[Plot] Saved KDE distributions to {save_path}")
    
    def plot_metrics_evolution(self):
        """Plot evolution of quality metrics (3 subplots)."""
        if not os.path.exists(self.metrics_csv_path):
            return
        
        df = pd.read_csv(self.metrics_csv_path)
        
        if df.empty:
            return
        
        # Save raw data for custom plotting
        data_path = os.path.join(self.data_dir, 'metrics_evolution_data.csv')
        df.to_csv(data_path, index=False)
        print(f"[Data] Saved metrics evolution data to {data_path}")
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Validity
        ax1 = axes[0]
        ax1.plot(df['epoch'], df['validity (%)'], marker='o', linewidth=2, color='blue')
        ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Validity (%)', fontsize=11, fontweight='bold')
        ax1.set_title('Molecular Validity Over Training', fontsize=12, fontweight='bold')
        ax1.tick_params(labelsize=10)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontweight('bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Uniqueness
        ax2 = axes[1]
        ax2.plot(df['epoch'], df['uniqueness (%)'], marker='s', linewidth=2, color='green')
        ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Uniqueness (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Molecular Uniqueness Over Training', fontsize=12, fontweight='bold')
        ax2.tick_params(labelsize=10)
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontweight('bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Diversity
        ax3 = axes[2]
        ax3.plot(df['epoch'], df['diversity'], marker='^', linewidth=2, color='red')
        ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Diversity', fontsize=11, fontweight='bold')
        ax3.set_title('Molecular Diversity Over Training', fontsize=12, fontweight='bold')
        ax3.tick_params(labelsize=10)
        for label in ax3.get_xticklabels() + ax3.get_yticklabels():
            label.set_fontweight('bold')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'metrics_evolution.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Plot] Saved metrics evolution to {save_path}")
    
    def generate_summary_report(self):
        """Generate a summary report of training."""
        if not os.path.exists(self.metrics_csv_path):
            return
        
        df = pd.read_csv(self.metrics_csv_path)
        
        if df.empty:
            return
        
        report_path = os.path.join(self.metrics_dir, 'training_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write("REINFORCE TRAINING SUMMARY REPORT\n")
            f.write("="*100 + "\n\n")
            
            f.write("Metric Definitions:\n")
            f.write("-"*100 + "\n")
            f.write("- Validity: % of generated molecules that are chemically valid\n")
            f.write("- Uniqueness: % of valid molecules that are unique (canonical SMILES)\n")
            f.write("- Diversity: Average Tanimoto distance between molecule pairs\n")
            f.write("\n")
            
            f.write("Overall Statistics:\n")
            f.write("-"*100 + "\n")
            f.write(f"Total epochs:             {len(df)}\n")
            f.write(f"Final validity:           {df['validity (%)'].iloc[-1]:.2f}%\n")
            f.write(f"Final uniqueness:         {df['uniqueness (%)'].iloc[-1]:.2f}%\n")
            f.write(f"Final diversity:          {df['diversity'].iloc[-1]:.4f}\n")
            f.write("\n")
            
            f.write("Property Statistics (Final Epoch):\n")
            f.write("-"*100 + "\n")
            f.write(f"Detonation Velocity:      {df['mean_detonation_velocity'].iloc[-1]:.2f} ± {df['std_detonation_velocity'].iloc[-1]:.2f} km/s\n")
            f.write(f"Density:                  {df['mean_density'].iloc[-1]:.2f} ± {df['std_density'].iloc[-1]:.2f} g/cm³\n")
            f.write(f"SA Score:                 {df['mean_sascore'].iloc[-1]:.2f} ± {df['std_sascore'].iloc[-1]:.2f}\n")
            f.write(f"Reward:                   {df['mean_reward'].iloc[-1]:.2f} ± {df['std_reward'].iloc[-1]:.2f}\n")
            f.write("\n")
            
            f.write("Best Performance:\n")
            f.write("-"*100 + "\n")
            best_validity_idx = df['validity (%)'].idxmax()
            f.write(f"Best validity:            {df['validity (%)'].iloc[best_validity_idx]:.2f}% (Epoch {df['epoch'].iloc[best_validity_idx]})\n")
            
            best_uniqueness_idx = df['uniqueness (%)'].idxmax()
            f.write(f"Best uniqueness:          {df['uniqueness (%)'].iloc[best_uniqueness_idx]:.2f}% (Epoch {df['epoch'].iloc[best_uniqueness_idx]})\n")
            
            best_diversity_idx = df['diversity'].idxmax()
            f.write(f"Best diversity:           {df['diversity'].iloc[best_diversity_idx]:.4f} (Epoch {df['epoch'].iloc[best_diversity_idx]})\n")
            
            best_reward_idx = df['mean_reward'].idxmax()
            f.write(f"Best reward:              {df['mean_reward'].iloc[best_reward_idx]:.2f} (Epoch {df['epoch'].iloc[best_reward_idx]})\n")
            
            f.write("\n")
            f.write("="*100 + "\n")
        
        print(f"[Report] Saved summary to {report_path}")
