import matplotlib.pyplot as plt

labels = ['Unmasked', 'Masked']
accuracy = [0.75, 0.70]

plt.figure(figsize=(7, 5))

# Better color for visualization
bars = plt.bar(labels, accuracy)

plt.ylabel('Accuracy')
plt.title('Accuracy Comparison (Masked vs Unmasked Faces)')
plt.ylim(0, 1)

# Add grid (improves readability)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add value labels on bars
for i, v in enumerate(accuracy):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

# Tight layout (prevents cut-off text)
plt.tight_layout()

plt.savefig("results/mask_vs_unmask.png", dpi=300)
plt.show()