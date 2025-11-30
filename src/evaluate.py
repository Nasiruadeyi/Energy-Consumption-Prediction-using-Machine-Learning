import matplotlib.pyplot as plt
import os

def save_custom_plot(fig, filename):
    """Save matplotlib figure to images folder."""
    if not os.path.exists("images"):
        os.makedirs("images")
    path = os.path.join("images", filename)
    fig.savefig(path)
