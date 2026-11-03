"""Build the simple, visual comparison notebook.

Run: uv run python scripts/build_multidetector_notebook.py
"""
from pathlib import Path

import nbformat as nb

cells = []


def md(text):
    cells.append(nb.v4.new_markdown_cell(text))


def code(text):
    cells.append(nb.v4.new_code_cell(
        text.strip(),
        metadata={
            'jupyter': {'source_hidden': True},
            'collapsed': True,
            'tags': ['hide-input'],
        },
    ))


md('''# How similar do these two pictures look?

On the left is the original picture. On the right is a modified copy made to look almost the same.

The percentage below compares how alike their shapes, light, and colors are. **100% means the pictures are identical.** A score close to 100% means they are very similar by this comparison; you may still notice differences.

You can read this page as it is. The pictures and result are already shown.''')

code('''from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from IPython.display import Markdown, display

ROOT = Path.cwd()
if not (ROOT / 'pyproject.toml').exists() and (ROOT.parent / 'pyproject.toml').exists():
    ROOT = ROOT.parent
OUT = ROOT / 'outputs' / 'multi_final'
original = np.array(Image.open(OUT / 'reference.png').convert('RGB'))
modified = np.array(Image.open(OUT / 'selection.png').convert('RGB'))
assert original.shape == modified.shape
similarity = float(structural_similarity(
    original, modified, channel_axis=2, data_range=255,
))
report = json.loads((OUT / 'report.json').read_text())
assert abs(similarity - report['selection']['ssim']) < 1e-9
height, width = original.shape[:2]
side = max(8, round(width / 4))
left = (width - side) // 2
upper = max(0, min(height - side, round(height * .30) - side // 2))
region = (slice(upper, upper + side), slice(left, left + side))


def show_comparison(first, second, highlight=False):
    fig, axes = plt.subplots(1, 2, figsize=(10, 7 if highlight else 5))
    for axis, picture, title in zip(
        axes, [first, second], ['Original image', 'Modified image'],
    ):
        axis.imshow(picture, interpolation='nearest')
        axis.set_title(title, fontsize=15, pad=12)
        if highlight:
            axis.add_patch(Rectangle(
                (left - .5, upper - .5), side, side,
                linewidth=2, edgecolor='#ffdb33', facecolor='none',
            ))
        axis.axis('off')
    plt.tight_layout()
    plt.show()


display(Markdown(f'## {similarity * 100:.2f}% visual similarity'))''')

md('''The two full pictures are shown at the same size. Compare the original on the left with the modified copy on the right.

**The yellow boxes mark the same small area in both pictures.** We will enlarge that area just below so you can take a closer look.''')

code('''show_comparison(original, modified, highlight=True)''')

md('''## A closer look at the same face

Below are the two areas inside the yellow boxes, enlarged to the same size: original on the left, modified copy on the right. Nothing has been added or made stronger to exaggerate the changes. We have only enlarged the pictures.

Look at the face, hair, and background to see whether you notice any differences. Small changes may be easier to see here than in the full pictures above. **The percentage above describes the full pictures**, not just this close-up.''')

code('''show_comparison(original[region], modified[region])''')

md('''The percentage describes how alike the pictures look in this comparison. It does not tell us whether Instagram or TikTok would treat them as the same image.''')

notebook = nb.v4.new_notebook(
    cells=cells,
    metadata={
        'kernelspec': {
            'name': 'python3',
            'display_name': 'Python 3 (uv)',
            'language': 'python',
        },
        'language_info': {'name': 'python'},
    },
)
nb.write(notebook, Path(__file__).resolve().parents[1] / 'multidetector_similarity.ipynb')
