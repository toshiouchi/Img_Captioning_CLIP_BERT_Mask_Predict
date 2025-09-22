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
## Generated captions

Generated captions for test data at epoch 10.

```
hypo: in this image we can see some strawberries the table surface.
refe: in this i can see there are red colored strawberries.
this pic. WER : 0.6666666666666666
this pic. BLEU: 0.6247616030574529
test number = 1 average, WER = 0.6666666865348816, BLEU = 0.6247615814208984
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/72e86a2a-7f1a-41ae-8bf5-b3771e7ab7c7.png)
　
 ```
hypo: in this image i can see few people sitting sitting on the..
refe: there are few persons sitting on the chairs. here we can see monitors, keyboards, tables, and devices.
this pic. WER : 0.8636363636363636
this pic. BLEU: 0.31947446576212973
test number = 2 average, WER = 0.7651515007019043, BLEU = 0.4721180200576782
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/ba295e00-927d-472c-9358-27bd90975243.png)
　
```
hypo: in this image we can see three persons standing on the the floor. buildings buildings.
refe: this image is taken outdoors. at the bottom of the there is a floor. in the background there are a few buildings with walls, windows and balconies. in the middle of the image two men and a woman are standing on the floor and they are with smiling faces.
this pic. WER : 0.8928571428571429
this pic. BLEU: 0.11606282923096954
test number = 3 average, WER = 0.8077200055122375, BLEU = 0.3534329831600189  
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/7ab532c2-7ec8-4e50-80dc-a78f4d6c0919.png)
　
 ```
hypo: in this image we can see some,, flowers, plants, sky the the clouds.
refe: in this image in front there are plants. in the background of the image there is sky.
this pic. WER : 0.7894736842105263
this pic. BLEU: 0.45279990304290557
test number = 4 average, WER = 0.8031584620475769, BLEU = 0.37827470898628235 
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/8f697685-d364-4234-8507-9f6e976e3f7f.png)
　
 ```
hypo: in this image we can see a woman wearing aggles.
refe: this is a black and white image. in this image we can see women wearing spectacles.
this pic. WER : 0.631578947368421
this pic. BLEU: 0.45330009958839473
test number = 5 average, WER = 0.7688425779342651, BLEU = 0.3932797908782959
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/c28eb13b-cdc8-4c2b-bd0b-fab54cc6853a.png)
　
 ```
hypo: in this image we can see some food in the plate plate.
refe: as we can see in the image there is a white color plate. in plate there is a dish.
this pic. WER : 0.8095238095238095
this pic. BLEU: 0.4792515333256459
test number = 6 average, WER = 0.7756227850914001, BLEU = 0.40760841965675354
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/29778bf9-1412-437c-8aef-a16887cfacc6.png)
　
 ```
hypo: in this image i can see a woman is standing and a guitar in a
refe: this image is clicked in a musical concert where there is a woman standing and she is holding a guitar in her hand. she is wearing black color dress. there is a mic in front of her and there is a bottle. she is holding a stick. there are speakers back side and there are some musical instruments on the bottom left corner.
this pic. WER : 0.8676470588235294
this pic. BLEU: 0.01311929651057275
test number = 7 average, WER = 0.7887691259384155, BLEU = 0.35125282406806946
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/27408da8-df9d-48e7-9919-6cd9de322462.png)
　
 ```
hypo: in this image we can see a dog.
refe: in this image, we can see a black color dog, there is a blurred background.
this pic. WER : 0.5
this pic. BLEU: 0.24153871270205204
test number = 8 average, WER = 0.7526729702949524, BLEU = 0.33753857016563416
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/9e5c5e7a-edd3-43c7-82ec-eef53715efcf.png)
　
 ```
hypo: in this image we can see food items on a plate.
refe: in this picture there is a bowl and a plate in the center of the image, which contains food items in it.
this pic. WER : 0.7916666666666666
this pic. BLEU: 0.2416736801845192
test number = 9 average, WER = 0.7570055723190308, BLEU = 0.326886922121048
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/f03fce68-d5c3-4d8e-a92d-5de5dad61d7f.png)
　
 ```
hypo: in this image we can see a food item the plate plate.
refe: in this picture i can observe some food places in the plate. the food is in brown, orange, green and red colors. it is looking like a burger. the background is completely blurred.
this pic. WER : 0.8205128205128205
this pic. BLEU: 0.07036031350037669
test number = 10 average, WER = 0.7633563280105591, BLEU = 0.30123424530029297
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/e99d4a70-9e0f-418c-9cfd-ab948431ce64.png)
　
 ```
hypo: in this image we can see a group standing people some floor and
refe: in this image i can see a person standing wearing a black shirt, blue jeans and glasses. he is holding a electronic gadget in his hand. in the background i can see few people standing, and the ceiling of the building.
this pic. WER : 0.8085106382978723
this pic. BLEU: 0.07395890302600679
test number = 11 average, WER = 0.7674612402915955, BLEU = 0.2805728614330292
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/b7a008a5-d5c3-4cc0-81e0-5889be4bb80e.png)
　
 ```
hypo: in this image we can see buildings, banners,,, banners,,,, sky and..
refe: in this image we can see cars, people, banners, hoardings, tent, pole, trees, boards, and buildings. in the background there is sky.
this pic. WER : 0.5294117647058824
this pic. BLEU: 0.345430104276125
test number = 12 average, WER = 0.7476237416267395, BLEU = 0.28597763180732727
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/0b66890d-1f53-4bd0-ade6-2cf276bb8879.png)
　
 ```
hypo: in this image we can see two standing standing and holding a gun. the we there are trees and and..
refe: in this image we can see two persons standing and holding the objects, there are some stones, grass, plants and trees, also we can see the sky.
this pic. WER : 0.625
this pic. BLEU: 0.5596540422693415
test number = 13 average, WER = 0.7381911873817444, BLEU = 0.307029664516449
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/789ec1ac-1aba-42af-a4b6-20ae28a5d90b.png)
　
 ```
hypo: in this image we can see a stone, plants, and and..
refe: in front of the image there are some engravings on the headstone, around the headstone on the surface there are green leaves and dry leaves and sticks, behind the headstone there are trees and a wall.
this pic. WER : 0.8571428571428571
this pic. BLEU: 0.03473874000754297
test number = 14 average, WER = 0.7466877102851868, BLEU = 0.28758031129837036
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/a54e4a4f-690b-4595-bd42-f1ca02c00fef.png)
　
```
hypo: in this image we can see a,, trees, sky and sky clouds.
refe: in this picture i can see building and few trees and a cloudy sky.
this pic. WER : 0.6
this pic. BLEU: 0.5413609128079863
test number = 15 average, WER = 0.7369085550308228, BLEU = 0.3044990003108978
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/f0bf2d78-df94-4de0-b80e-479bf4609ed4.png)
　
```
hypo: in this image we can see a group of people standing and holding the background we there some buildings.
refe: in this image there are many people in front of the building. some of them are holding camera. in the background there are buildings. there is a banner over here.
this pic. WER : 0.7352941176470589
this pic. BLEU: 0.4571905521725783
test number = 16 average, WER = 0.7368077039718628, BLEU = 0.31404221057891846
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/ea053545-4da2-4732-a8ed-0dd0ca5b8e6f.png)
　
```
hypo: in this image we can see a snake on the and...
refe: in this image i can see a snake on the ground. it is in black color. i can see few wooden sticks, few stones and grass.
this pic. WER : 0.6
this pic. BLEU: 0.20100947378845824
test number = 17 average, WER = 0.7287602424621582, BLEU = 0.30739325284957886
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/51e0639b-ff3d-4814-b85a-0b241f0b1844.png)
　
```
hypo: in this image we can see a woman standing and holding a paper. her there is see a..
refe: in this image we can see a person standing and holding a book and to the side we can see a podium with mic and there is a laptop and some other objects on the table. we can see a person standing in the bottom right.
this pic. WER : 0.6666666666666666
this pic. BLEU: 0.19437398934385508
test number = 18 average, WER = 0.7253105640411377, BLEU = 0.30111441016197205
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/4f6ad1e7-6036-47f2-b89e-4a32bdc9d0f2.png)
　
```
hypo: in this image we can see two women.
refe: in this picture i can see 2 women in front and the women right is holding a brush in her hand and i see the paint on the face of the woman on the left and in the background i see the grass and on the top left of this image i see the blue color things.
this pic. WER : 0.896551724137931
this pic. BLEU: 0.0019152827775531266
test number = 19 average, WER = 0.7343232035636902, BLEU = 0.28536707162857056
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/f5d1768d-cb54-4c3d-90ed-a2509e5bd1b3.png)
　
```
hypo: in this image we can see a bear bear and the water.
refe: in this image we can see an animal, water, rocks, and leaves. at the bottom of the image we can see a person who is truncated.
this pic. WER : 0.7096774193548387
this pic. BLEU: 0.2028533308805255
test number = 20 average, WER = 0.7330909967422485, BLEU = 0.28124135732650757
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/d7423c3e-6202-4141-a8e8-11da69634d24.png)
　
```
hypo: in this image we can see three standing standing on the ground,.
refe: in this image we can see three people, one of them is wearing a backpack, in front of them, we can see some bags, box, also we can see some plants, grass, and trees.
this pic. WER : 0.7857142857142857
this pic. BLEU: 0.16172690339062004
test number = 21 average, WER = 0.735596776008606, BLEU = 0.2755502164363861
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/3f145d18-638c-47a7-8f68-f55602a60612.png)
　
```
test 21 average WER : 0.7355967920920637
test 21 average BLEU: 0.2755502224593148
```
