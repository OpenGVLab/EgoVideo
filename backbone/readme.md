# Backbone Model For EgoVideo

A sample codebase for visual/text extraction and evaluation

## Installation
For installation, you can run

```
pip install -r requirements.txt
```

## Feature Extraction
If you want to extract features by our model, you can easily run the following code:
```python
import torch

from model.setup_model import *
import argparse
import pandas as pd

model,tokenizer = build_model(ckpt_path = 'ckpt_4frames.pth',num_frames = 4)
model = model.eval().to('cuda').to(torch.float16)
vision_input = torch.randn(1,3,4,224,224).to('cuda') #[B,C,T,H,W]
text = 'I want to watch a movie.'
text = tokenizer(text,max_length=20,truncation=True,padding = 'max_length',return_tensors = 'pt')
text_input = text.input_ids.to('cuda')
mask = text.attention_mask.to('cuda')
image_features, text_features = model(vision_input,text_input,mask)
print(image_features.shape) # [1,512]
print(text_features.shape) #[1,512]
```

## Evaluation for Zero-Shot on Ek-100 mir dataset
You can easily run [`eval_ek100_mir.py`](eval_ek100_mir.py) for evaluation. You can change your own config in [`configs/demo_ek100_mir.yaml`](configs/demo_ek100_mir.yaml). If you use our checkpoint, you will get `0.459` on mAP and `0.386` on nDCG. 
