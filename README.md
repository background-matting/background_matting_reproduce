# background_matting_reproduction

>This is the reproduction of Background Matting: The World is Your Green Screen

Our code focus on picture matting which mainly contains four parts(the experiment is two stage):
* module encoder part: Jiang Di
* module decoder part: Bian Xinhui
* module discriminator part: Li Danyang
* training and testing part: all three numbers
* readme and code organized: Bian Xinhui

The detailed development is added on the top of each python file.
* * *
## Dataset

We use two Datasets here, The first is <i>Adobe Deep Image Matting Dataset</i>, the second is the sample our reproduced paper provided. The authors provide some train videos and test videos, we cut those video by frames to get our second stage training and testing dataset.

Follow the <a href="https://sites.google.com/view/deepimagematting">instruction</a> to contact author for the first dataset.


* * *
## Code
`calculate.py` could calculate the MSE,SAD and IoU of our alpha matte and ground truth.
>We put two samples in alpha document, you could easily run the following code to get the result.
<pre><code>python calculate.py -op alpha/GAN_alpha0.png -gt alpha/GT.png -ops alpha/GAN_alpha0.pt -gts alpha/GT.pt</code></pre>

`bayesian_matting.py` is our compared module, Thanks for <a href="https://github.com/MarcoForte/bayesian-matting">MarcoForte's</a> work on github and <a href="https://github.com/SamuelYG/trimap_generate">yanggang's</a> trimap generate method. We reuse their code to get the bayesian result.

`Composition_code.py` is provided by Adobe, we modified it to achieve generating new images with different background parts. `train_Adobe.py`,`train_withGAN.py`,`test_adobe.py`,`test_real.py` and `test_one_file.py` use this python script.

`CSblock.py` is one of the novel part of the paper. It is part of the whole model in `model.py`

`decoder.py` is another part of the paper. It is part of the whole model in `model.py`.

`discriminator.py` is the part in GAN training. it is called in  `train_withGAN.py`

`loss_function.py` only contains for loss functions: L1 loss; compose loss; gradient loss and GANloss. The combination of those loss could calculate the GAN loss in `train_withGAN.py`.

`model.py` is the generate model, encoder and encoder is used in it.

`oritin_data_loader.py` contains all data preprocessors (Adobe image and video frame cut).

`save_black_img.py` generate a black background. it is used in visualized foreground result in `test_adobe.py`

`util.py` contains three methods, which helps convert tensor to image and convert image to tensor

##### below are the train and test scripts

`train_Adobe.py` trains the first stage. Because the model is huge, we have tested that it should run at least in two GPUs and using dataParallel. We create a document called net to save the checkpoint. Now you could run the following code to have a look at the implementation(if you have the data).
<pre><code>python train_Adobe.py -np net/net.pt -op net/opt.pt</code></pre>

`test_Adobe.py` could test the result of the first stage. We put two checkpoints in net document and you could run following code to see the result(for convenience We only output the first result)(if you have the data).
<pre><code>python test_Adobe.py -np net/net_10.pt -op net/opt_10.pt</code></pre>

`train_withGAN.py` This GAN structure is very huge, we test that we need to use at least three GPUs to run. And we haven't find the best model until now(time and data problem). If you want to run, just type the following code in command(if you have the data).
<pre><code>python train_withGAN.py </code></pre>

`test_one_file.py` I put one group of data and you could visualize the result in one_img file after running the code.
<pre><code>python test_one_file.py -np net/net_10.pt </code></pre>
* * *

## Reference

1. Ning Xu, Brian Price, Scott Cohen, Thomas Huang.  Deep Image Matting.  In Proceedings of the Conference on Computer Vision and Pattern Recognition (CVPR) 2017.
2. Yung-Yu Chuang, Brian Curless, David H. Salesin, and Richard Szeliski. A Bayesian Approach to Digital Matting. In Proceedings of IEEE Computer Vision and Pattern Recognition (CVPR 2001), Vol. II, 264-271, December 2001
3. Sengupta S, Jayaram V, Curless B, et al. Background matting: The world is your green screen[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020: 2291-2300
4. P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1125–1134, 2017
