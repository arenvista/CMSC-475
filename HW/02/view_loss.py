import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from prompt_toolkit import prompt
from prompt_toolkit.completion import FuzzyWordCompleter
from pathlib import Path

def create_out_dir(id):
    csv_dir = "output/img/" + id + "/"
    csv_dir_path = Path(csv_dir)
    csv_dir_path.mkdir(parents=True, exist_ok=True)
    return csv_dir_path
    


def get_all_files(directory):
    """Recursively fetches all files in the given directory."""
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, directory)
            file_list.append(relative_path)
    return file_list


def fuzzy_find_file(directory_path="."):
    print(f"Scanning directory: {directory_path}...")
    files = get_all_files(directory_path)

    if not files:
        print("No files found in the specified directory.")
        return None

    # Create a fuzzy completer with our list of files
    completer = FuzzyWordCompleter(files)

    try:
        # Launch the interactive prompt
        selected_file = prompt(
            "Type to fuzzy search (Tab to complete) > ", completer=completer
        )

        # Verify the user didn't just type gibberish that isn't a file
        if selected_file in files:
            full_selected_path = os.path.join(directory_path, selected_file)
            print(f"\nSuccess! You selected: {full_selected_path}")
            return full_selected_path
        else:
            print("\nInvalid selection.")
            return None

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nOperation cancelled by user.")
        return None


def plot_normalized_loss_agg(data, loss_columns=None, save_path='my_pretty_lineplot.png'):
    """
    Normalizes and plots model training loss data over iterations.

    Args:
        data (pd.DataFrame): The dataframe containing the loss data and an 'itter' column.
        loss_columns (list): List of column names to plot. Defaults to standard loss columns.
        save_path (str): The filename/path to save the resulting plot.
    """
    if loss_columns is None:
        loss_columns = ["loss_current", "loss_average", "running_loss"]

    # Use a copy so we don't overwrite the original dataframe's values
    plot_data = data.copy()

    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

    # --- 1. Data Prep & Normalization ---
    # Capture Original Min/Max for the text box
    stats_text = "Original Min/Max Values:\n\n"
    for col in loss_columns:
        orig_min = plot_data[col].min()
        orig_max = plot_data[col].max()
        stats_text += f"• {col}:\n  {orig_min:.4f} / {orig_max:.4f}\n"

    # Normalize the data between 0 and 1
    for col in loss_columns:
        min_val = plot_data[col].min()
        max_val = plot_data[col].max()
        
        # Guard against division by zero if max and min are identical
        if max_val != min_val:
            plot_data[col] = (plot_data[col] - min_val) / (max_val - min_val)
        else:
            plot_data[col] = 0

    # Reshape the normalized data
    melted_data = pd.melt(
        plot_data,
        id_vars=["itter"], 
        value_vars=loss_columns,
        var_name="Loss Type",  
        value_name="Normalized Loss" 
    )

    # --- 2. Create the Plot with Enhanced Sizing ---
    fig, ax = plt.subplots(figsize=(10, 6)) 

    # Plot with thicker lines
    sns.lineplot(
        data=melted_data,
        x="itter",
        y="Normalized Loss",
        hue="Loss Type",
        linewidth=1.5,
        ax=ax
    )

    # --- 3. Style the Labels and Title ---
    plt.xlabel("Iterations", fontweight='bold', labelpad=12)
    plt.ylabel("Normalized Loss", fontweight='bold', labelpad=12)
    plt.title("Model Training Loss Over Iterations", fontsize=16, fontweight='black', pad=20)

    # --- 4. Style the Legend ---
    plt.legend(
        title="Loss Function", 
        title_fontsize='12',
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        borderaxespad=0.,
        frameon=True,
        shadow=True 
    )

    # --- 5. Style the Text Box ---
    plt.text(
        x=1.03, 
        y=0.45, 
        s=stats_text, 
        transform=ax.transAxes, 
        fontsize=9, 
        verticalalignment='top', 
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', alpha=0.9, edgecolor='#ced4da') 
    )

    # --- 6. Final Cleanup & Save ---
    sns.despine(left=True, bottom=True) 

    plt.tight_layout() 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # Free memory
    
    print(f"Plot saved successfully to: {save_path}")

# ==========================================
# Example usage:
# ==========================================
# df = pd.read_csv('my_training_data.csv')



def plot_single(data, loss_columns=None, save_path='my_pretty_lineplot.png'):
    """
    Normalizes and plots model training loss data over iterations.

    Args:
        data (pd.DataFrame): The dataframe containing the loss data and an 'itter' column.
        loss_columns (list): List of column names to plot. Defaults to standard loss columns.
        save_path (str): The filename/path to save the resulting plot.
    """
    if loss_columns is None:
        loss_columns = ["loss_current", "loss_average", "running_loss"]

    # Use a copy so we don't overwrite the original dataframe's values
    plot_data = data.copy()

    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

    # Reshape the normalized data
    melted_data = pd.melt(
        plot_data,
        id_vars=["itter"], 
        value_vars=loss_columns,
        var_name="Loss Type",  
        value_name="Raw Loss" 
    )

    # --- 2. Create the Plot with Enhanced Sizing ---
    fig, ax = plt.subplots(figsize=(10, 6)) 

    # Plot with thicker lines
    sns.lineplot(
        data=melted_data,
        x="itter",
        y="Raw Loss",
        hue="Loss Type",
        linewidth=1.5,
        ax=ax
    )

    # --- 3. Style the Labels and Title ---
    plt.xlabel("Iterations", fontweight='bold', labelpad=12)
    plt.ylabel("Raw Loss", fontweight='bold', labelpad=12)
    plt.title("Model Training Loss Over Iterations", fontsize=16, fontweight='black', pad=20)

    # --- 4. Style the Legend ---
    plt.legend(
        title="Loss Function", 
        title_fontsize='12',
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        borderaxespad=0.,
        frameon=True,
        shadow=True 
    )

    # --- 6. Final Cleanup & Save ---
    sns.despine(left=True, bottom=True) 

    plt.tight_layout() 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # Free memory
    
    print(f"Plot saved successfully to: {save_path}")


if __name__ == "__main__":
    target_dir = os.path.abspath("./data/loss")
    path = fuzzy_find_file(target_dir)
    if not path: raise ValueError("Must Enter Path")
    df = pd.read_csv(path)
    id = Path(path).stem
    out_path_dir = create_out_dir(id)
    plot_normalized_loss_agg(df, save_path=f'{out_path_dir}/final_model_loss_agg.png')
    loss_columns = ["loss_current", "loss_average", "running_loss"]
    for l in loss_columns:
        plot_single(df,[l],f"{out_path_dir}/{l}.png" )
