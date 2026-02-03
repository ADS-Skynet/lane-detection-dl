# Run this to inspect the checkpoint file
# python3 -c
import torch
checkpoint = torch.load('/home/siwoo/finetuned_best.pth', map_location='cpu', weights_only=False)
print('Checkpoint keys:', checkpoint.keys())
if 'model' in checkpoint:
    print('Number of model weights:', len(checkpoint['model']))
    print('First 5 keys:', list(checkpoint['model'].keys())[:5])
elif 'state_dict' in checkpoint:
    print('Number of model weights:', len(checkpoint['state_dict']))
    print('First 5 keys:', list(checkpoint['state_dict'].keys())[:5])
else:
    print('Number of weights:', len(checkpoint))
    print('First 5 keys:', list(checkpoint.keys())[:5])