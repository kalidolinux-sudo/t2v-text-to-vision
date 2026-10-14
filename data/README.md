# Data

| File | What it is | Made by |
|---|---|---|
| `imagenet_names.json` | short, unique names of the 1,000 ImageNet-1k classes | `t2v.data.imagenet_names` |
| `imagenet_split.json` | the fixed class split: 900 seen, 49 random held-out, 51 held-out as whole CLIP clusters | `t2v.data.split_classes` |
| `descriptions_imagenet.json` | 5 visual descriptions for each of the 1,000 classes | the agent on AMD TokenFactory (`t2v.describe`) |
| `neighbors_imagenet.json` | visual analogies for the 100 held-out classes and 100 validation classes | the agent on AMD TokenFactory (`t2v.neighbors`) |
| `descriptions.json` | 5 visual descriptions for each CIFAR-100 class | the agent on AMD TokenFactory |

Images are not stored here: CIFAR-100 and ImageNet-1k 32×32 are downloaded from Hugging Face on first use
and cached in this folder as `*.pt` files (git-ignored).
