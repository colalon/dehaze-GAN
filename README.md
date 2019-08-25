<h1 id="dehaze-GAN"> dehaze-GAN</h1>

*   [演算法說明](#Algo)
    *   [除霧架構](#Architecture)
    *   [生成器](#generator)
    *   [分辨器](#discriminator)
    *   [training epoachs](#training-epoachs)
*   [使用方法](#howtouse)
    *   [訓練](#training)
    *   [測試](#testing)
    *   [圖片除霧](#images_test)
    *   [影片除霧](#video_test)
*   [結果展示](#result)
    *   [圖片除霧結果](#images_result)
    *   [影片除霧結果](#video_result)

<h2 id="Algo"> 演算法說明</h2>
<h3 id="Architecture"> 除霧架構</h3>
<img src="./doc/Architecture.jpg" width="700px"/>

<h3 id="generator"> 生成器</h3>

| Unet1 | Unet2 |
| --- | --- |
| <img src="./doc/Unet.jpg" width="500px"/> | <img src="./doc/Unet2.jpg" width="500px"/> |

<h3 id="discriminator"> 分辨器</h3>
<img src="./doc/discriminator.jpg" width="500px"/>

<h3 id="training-epoachs"> training epoachs</h3>
<img src="./doc/training epochs.jpg" width="700px"/>

<h2 id="howtouse"> 使用方法</h2>

<h3 id="training"> 訓練</h3>

```sh
python main.py --phase train --pretrain (True or False) --weights_path ./weights_dir/weights.ckpt --dataset_path_x ./dataset_x.npy --dataset_path_y ./dataset_y.npy  --output_dir ./output/
```
<h3 id="testing"> 測試</h3>

```sh
python main.py  --phase test --pretrain True --weights_path ./weights_dir/weights.ckpt --dataset_path_x ./dataset_x.npy --dataset_path_y ./dataset_y.npy  --output_dir ./output/
```
<h3 id="images_test"> 圖片除霧</h3>

```sh
python main.py  --phase images --pretrain True --weights_path ./weights_dir/weights.ckpt --imlist ["./sample1.jpg","./sample2.png"] --output_dir ./output/
```
<h3 id="video_test"> 影片除霧</h3>

```sh
python main.py  --phase video --pretrain True --weights_path ./weights_dir/weights.ckpt --video_path ./video.avi --output_dir ./output/
```

<h2 id="result"> 結果展示</h2>

<h3 id="images_result"> 圖片除霧結果</h3>

| input | output | input | output |
| --- | --- | --- | --- |
|<img src="./doc/land/input1.jpg" width="200px"/>|<img src="./doc/land/output1.jpg" width="200px"/>|<img src="./doc/underwater/input1.jpg" width="200px"/>|<img src="./doc/underwater/output1.jpg" width="200px"/>|
|<img src="./doc/land/input2.jpg" width="200px"/>|<img src="./doc/land/output2.jpg" width="200px"/>|<img src="./doc/underwater/input2.jpg" width="200px"/>|<img src="./doc/underwater/output2.jpg" width="200px"/>|
|<img src="./doc/land/input3.jpg" width="200px"/>|<img src="./doc/land/output3.jpg" width="200px"/>|<img src="./doc/underwater/input3.jpg" width="200px"/>|<img src="./doc/underwater/output3.jpg" width="200px"/>|
|<img src="./doc/land/input4.jpg" width="200px"/>|<img src="./doc/land/output4.jpg" width="200px"/>|<img src="./doc/underwater/input4.jpg" width="200px"/>|<img src="./doc/underwater/output4.jpg" width="200px"/>|
|<img src="./doc/land/input5.jpg" width="200px"/>|<img src="./doc/land/output5.jpg" width="200px"/>|<img src="./doc/underwater/input5.jpg" width="200px"/>|<img src="./doc/underwater/output5.jpg" width="200px"/>|
|<img src="./doc/land/input6.jpg" width="200px"/>|<img src="./doc/land/output6.jpg" width="200px"/>|<img src="./doc/underwater/input6.jpg" width="200px"/>|<img src="./doc/underwater/output6.jpg" width="200px"/>|

<h3 id="video_result"> 影片除霧結果</h3>

