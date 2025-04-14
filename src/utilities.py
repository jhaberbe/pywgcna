import numpy as np
import seaborn as sns
from gprofiler import GProfiler

def labels_to_hex_colors(labels):
    unique_labels = np.unique(labels)
    color_map = sns.color_palette("hsv", len(unique_labels))
    label_to_color = {label: f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" 
                      for label, (r, g, b) in zip(unique_labels, color_map)}
    return [label_to_color[label] for label in labels]

def get_gprofiler_results(gene_labels, organism='hsapiens'):
    """
    Fetch gProfiler results for a given list of gene labels.

    Parameters:
        gene_labels (list): List of gene labels.
        organism (str): Organism name (default is 'hsapiens' for human).

    Returns:
        pandas.DataFrame: DataFrame containing gProfiler results.
    """
    gp = GProfiler(return_dataframe=True)
    results = gp.profile(organism=organism, query=gene_labels)
    return results