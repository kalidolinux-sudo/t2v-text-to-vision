# Upgrade tests after T2V-1K

Colab notebooks that test candidate upgrades of the shipped T2V-1K. Each one chooses its settings by
cross-validation on seen classes treated as new, then scores the official test tasks once against the
shipped model. Nothing here is used by the product unless the results below say so.

| Notebook | What it tests | Result | Shipped |
|---|---|---|---|
| [`T2V_rowmap_upgrade.ipynb`](T2V_rowmap_upgrade.ipynb) | Classifier rows of new classes from a text → row map trained on photos of the 900 seen classes, with the TokenFactory descriptions and a 60-attribute TokenFactory questionnaire as input | half new +2.3, all new +0.8, hard −1.9 points ([log](../../results/upgrade_tests/rowmap_attributes.txt)) | no |
| [`T2V_analogies_v2.ipynb`](T2V_analogies_v2.ipynb) | TokenFactory analogies v2: 40 described candidates, 32×32 visual criteria, three answers averaged (needs `TOKENFACTORY_API_KEY`) | not measured yet | no |

Both need a T4 GPU (*Runtime → Change runtime type*).
