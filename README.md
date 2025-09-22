We have fine-tuned with image captioning system by mask prediction with length predictor and by mask prediction without length predictor using PAD token. System consists of clip, dense connector and bert.
V7 dataset was used with fine-tuning. A system without the length predictor generated better captions than a system with the length predictor. We report essense of a system without length predictor.

Program using measurement is in without_length_predictor folder.

# Report of fine-tuning image captioning system with clip, dense connector and bert by mask prediction without length predictor.

## For stablity of fine-tuning

### Monitor the parameter gradients of Clip's upstream layer and Bert's downstream layer.

```python
#for name, param in model.named_parameters():
#    print( name )
            
norm0 = torch.sqrt( torch.norm( model.clip_model.vision_model.encoder.layers[0].self_attn.q_proj.weight.grad, p = 2 ) ).item()
norm1 = torch.sqrt( torch.norm( model.bert.encoder.layer[23].attention.self.query.weight.grad, p = 2 ) ).item()
norm_mean = torch.mean( torch.stack ([ torch.sqrt( torch.norm( param.grad, p = 2 ) ) for param in model.parameters() if param.grad is not None ] ) ).item()
with open(norm_file, 'a') as f:
    print( "epcoch:", epoch, ", step:", global_step, ", norm0:", norm0, ", norm1:", norm1, ", norm_mean:", norm_mean, file=f  )
    f.flush()
```

### Setting a small learning rate

```
clip:   2e-7
bert:   2e-5
others: 1e-4
```

### Using the Learning Rate Schedule

```python
cheduler = get_linear_schedule_with_warmup( optimizer, num_warmup_steps, num_global_steps )
```
Training was for 10 epochs, with batch_size = 20 and len( train_lodaer ) = 20298. The learning rate increased linearly from 0 to the set learning rate for the first epoch, then decreased to 0 over the next 9 epochs.

### AdamW, weight_decay and betas settings

```python
params_clip = []
params_bert = []
params_others = []
for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        if 'clip_model' in name:
            params_clip.append(parameter)
        elif 'bert' in name:
            params_bert.append(parameter)
        else:
            params_others.append(parameter)
param_groups = [
    {'params': params_clip, 'lr': 2e-7},
    {'params': params_bert, 'lr': 2e-5},
    {'params': params_others, 'lr': 1e-4}
]

optimizer = torch.optim.AdamW( param_groups, weight_decay = 0.01, betas= (0.9, 0.999) )
```
### Not using grad_clip

## Using AMP and Scaler for calculation speed.

## Mask prediction without length predictor using PAD token.

### Do not specify padding_idx in nn.Embedding.

### Do not specify tokenizer.pad_token_id in ignore_index in cross-entropy loss.

### When applying a mask to the teacher caption, include the pad position.

### Do not use the attention mask in padding.

### Use PAD to make the length of all teacher captions and masked captions a fixed length. Start with this fixed-length MASK caption when inferencing.

## Loss, WER, BLEU

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/8a447bf9-7991-489a-948d-960d8b38acbd.png)

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/70e381eb-29ae-4e18-a764-bdfc90f42e49.png)

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/e4470c2a-b9ca-4d97-9ff6-84934e5b953d.png)


### Train and Val
10 epochs
```
       loss   WER  BLEU
train  1.18  39.2  74.8
val    1.11  37.9  75.1
```
