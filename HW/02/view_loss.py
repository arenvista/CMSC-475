import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  # Updated from pylab to pyplot
from prompt_toolkit import prompt
from prompt_toolkit.completion import FuzzyWordCompleter
from pathlib import Path
from typing import List, Optional, Union

def create_out_dir(identifier: str) -> Path:
    """Creates the output directory for a given identifier."""
    # Using purely pathlib for cleaner syntax
    out_dir = Path("output/img") / identifier
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def get_all_files(directory: Union[str, Path]) -> List[str]:
    """Recursively fetches all file paths relative to the given directory."""
    directory_path = Path(directory)
    # Using pathlib's rglob for a more modern recursive search than os.walk
    return [
        str(p.relative_to(directory_path))
        for p in directory_path.rglob("*")
        if p.is_file()
    ]

def fuzzy_find_file(directory_path: Union[str, Path] = ".") -> Optional[Path]:
    """Prompts the user to fuzzy search for a file in the directory."""
    print(f"Scanning directory: {directory_path}...")
    files = get_all_files(directory_path)

    if not files:
        print("No files found in the specified directory.")
        return None

    completer = FuzzyWordCompleter(files)

    try:
        selected_file = prompt(
            "Type to fuzzy search (Tab to complete) > ", completer=completer
        )

        if selected_file in files:
            full_selected_path = Path(directory_path) / selected_file
            print(f"\nSuccess! You selected: {full_selected_path}")
            return full_selected_path
        else:
            print("\nInvalid selection.")
            return None

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return None

def plot_loss(
    data: pd.DataFrame,
    loss_columns: List[str],
    save_path: Path,
    normalize: bool = False,
    y_label: str = "Raw Loss"
):
    """
    Plots model training loss data over iterations. Consolidates both 
    normalized aggregate plotting and raw single-line plotting.
    """
    plot_data = data.copy()
    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

    stats_text = "Original Min/Max Values:\n\n" if normalize else ""

    # 1. Data Prep & Normalization (Single Pass)
    if normalize:
        y_label = "Normalized Loss"
        for col in loss_columns:
            min_val = plot_data[col].min()
            max_val = plot_data[col].max()
            
            # Capture for text box
            stats_text += f"• {col}:\n  {min_val:.4f} / {max_val:.4f}\n"

            # Apply math
            if max_val != min_val:
                plot_data[col] = (plot_data[col] - min_val) / (max_val - min_val)
            else:
                plot_data[col] = 0

    # 2. Reshape data
    melted_data = pd.melt(
        plot_data,
        id_vars=["itter"], # NOTE: Assuming "itter" is correctly misspelled in your CSV
        value_vars=loss_columns,
        var_name="Loss Type",
        value_name=y_label
    )

    # 3. Create Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=melted_data,
        x="itter",
        y=y_label,
        hue="Loss Type" if len(loss_columns) > 1 else None, # Skip hue if it's a single plot
        linewidth=1.5,
        ax=ax
    )

    # 4. Style Labels and Title
    plt.xlabel("Iterations", fontweight='bold', labelpad=12)
    plt.ylabel(y_label, fontweight='bold', labelpad=12)
    plt.title("Model Training Loss Over Iterations", fontsize=16, fontweight='black', pad=20)

    # 5. Style Legend (Only if multiple lines exist)
    if len(loss_columns) > 1:
        plt.legend(
            title="Loss Function",
            title_fontsize='12',
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            borderaxespad=0.,
            frameon=True,
            shadow=True
        )

    # 6. Append Text Box (Only if normalized)
    if normalize:
        plt.text(
            x=1.03,
            y=0.45,
            s=stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', alpha=0.9, edgecolor='#ced4da')
        )

    # 7. Cleanup & Save
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    # Save using the pathlib object
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig) 
    print(f"Plot saved successfully to: {save_path}")


if __name__ == "__main__":
    target_dir = Path("./data/loss").resolve()
    
    # Ensure target directory exists before scanning
    target_dir.mkdir(parents=True, exist_ok=True) 
    
    path = fuzzy_find_file(target_dir)
    
    if not path: 
        raise ValueError("Must Enter Path")
        
    df = pd.read_csv(path)
    identifier = path.stem
    out_path_dir = create_out_dir(identifier)
    
    default_loss_columns = ["loss_current", "loss_average", "running_loss"]
    
    # Safety Check: Ensure the columns actually exist in the CSV to prevent KeyErrors
    valid_cols = [col for col in default_loss_columns if col in df.columns]
    if not valid_cols:
        raise ValueError("None of the expected loss columns were found in the CSV.")

    # Generate the normalized aggregate plot
    plot_loss(
        data=df, 
        loss_columns=valid_cols, 
        save_path=out_path_dir / 'final_model_loss_agg.png',
        normalize=True
    )
    
    # Generate the individual raw plots
    for l in valid_cols:
        plot_loss(
            data=df, 
            loss_columns=[l], 
            save_path=out_path_dir / f"{l}.png",
            normalize=False
        )
