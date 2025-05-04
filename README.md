<div align="center">

<h1>In-Context Edit: Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer</h1>

<div>
    <a href="https://river-zhang.github.io/zechuanzhang//" target="_blank">Zechuan Zhang</a>&emsp;
    <a href="https://horizonwind2004.github.io/" target="_blank">Ji Xie</a>&emsp;
    <a href="https://yulu.net.cn/" target="_blank">Yu Lu</a>&emsp;
    <a href="https://z-x-yang.github.io/" target="_blank">Zongxin Yang</a>&emsp;
    <a href="https://scholar.google.com/citations?user=RMSuNFwAAAAJ&hl=zh-CN&oi=ao" target="_blank">Yi Yang✉</a>&emsp;
</div>
<div>
    ReLER, CCAI, Zhejiang University; Harvard University
</div>
<div>
     <sup>✉</sup>Corresponding Author
</div>
<div>
    <a href="https://arxiv.org/abs/2504.20690" target="_blank">Arxiv</a>&emsp;
    <a href="https://huggingface.co/spaces/RiverZ/ICEdit" target="_blank">Huggingface Demo 🤗</a>&emsp;
    <a href="https://river-zhang.github.io/ICEdit-gh-pages/" target="_blank">Project Page</a>
</div>


<div style="width: 80%; margin:auto;">
    <img style="width:100%; display: block; margin: auto;" src="docs/images/teaser.png">
    <p style="text-align: left;"><strong>Image Editing is worth a single LoRA!</strong> We present In-Context Edit, a novel approach that achieves state-of-the-art instruction-based editing <b>using just 0.5% of the training data and 1% of the parameters required by prior SOTA methods</b>. The first row illustrates a series of multi-turn edits, executed with high precision, while the second and third rows highlight diverse, visually impressive single-turn editing results from our method.</p>
</div>

:open_book: For more visual results, go checkout our <a href="https://river-zhang.github.io/ICEdit-gh-pages/" target="_blank">project page</a>

This repository will contain the official implementation of _ICEdit_.


<div align="left">

# 📖 Table of Contents

1. [⚠️ Tips](#⚠️-tips)
2. [🎆 News](#-news)
3. [💼 Installation](#-installation)
   - [Conda environment setup](#conda-environment-setup)
   - [Download pretrained weights](#download-pretrained-weights)
   - [Inference in bash (w/o VLM Inference-time Scaling)](#inference-in-bash-wo-vlm-inference-time-scaling)
   - [Inference in Gradio Demo](#inference-in-gradio-demo)
   - [ComfyUI Workflow](#comfyui-workflow)
4. [Comparison with Commercial Models](#comparison-with-commercial-models)
5. [Star History](#star-history)
6. [Bibtex](#bibtex)


# ⚠️ Tips

### If you encounter such a failure case, please **try again with a different seed**!

- Our base model, FLUX, does not inherently support a wide range of styles, so a large portion of our dataset involves style transfer. As a result, the model **may sometimes inexplicably change your artistic style**.

- Our training dataset is **mostly targeted at realistic images**. For non-realistic images, such as **anime** or **blurry pictures**, the success rate of the editing **drop and could potentially affect the final image quality**.

- While the success rates for adding objects, modifying color attributes, applying style transfer, and changing backgrounds are high, the success rate for object removal is relatively lower due to the low quality of the removal dataset we use.

The current model is the one used in the experiments in the paper, trained with only 4 A800 GPUs (total `batch_size` = 2 x 2 x 4 = 16). In the future, we will enhance the dataset, and do scale-up, finally release a more powerful model.

# To Do List

- [x] Inference Code
- [ ] Inference-time Scaling with VLM
- [x] Pretrained Weights
- [ ] More Inference Demos
- [x] Gradio demo
- [x] Comfy UI demo (by @[judian17](https://github.com/River-Zhang/ICEdit/issues/1#issuecomment-2846568411), compatible with [nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku), support high-res refinement and FLUX Redux)
- [ ] Comfy UI demo official
- [ ] Training Code

# 🎆 News 
- **[2025/5/3]** 🔥 We extend our heartfelt gratitude to [softicelee2](https://github.com/River-Zhang/ICEdit/issues/6#issue-3037410834) for creating an outstanding Chinese tutorial [Youtube video](https://www.youtube.com/watch?v=rRMc5DE4qMo) on using ICEdit! 
- **[2025/5/2]** 🌟 Heartfelt thanks to [judian17](https://github.com/River-Zhang/ICEdit/issues/1#issuecomment-2846568411) for crafting an amazing [Comfy UI demo](https://github.com/River-Zhang/ICEdit/issues/1#issuecomment-2846568411)! 🚀 Dive in and give it a spin!
- **[2025/4/30]** 🔥 We release the [Huggingface Demo](https://huggingface.co/spaces/RiverZ/ICEdit) 🤗! Have a try!
- **[2025/4/30]** 🔥 We release the [paper](https://arxiv.org/abs/2504.20690) on arXiv!
- **[2025/4/29]** We release the [project page](https://river-zhang.github.io/ICEdit-gh-pages/) and demo video! Codes will be made available in next week~ Happy Labor Day!


# 💼 Installation

## Conda environment setup

```bash
conda create -n icedit python=3.10
conda activate icedit
pip install -r requirements.txt
pip install -U huggingface_hub
```

## Download pretrained weights

If you can connect to Huggingface, you don't need to download the weights. Otherwise, you need to download the weights to local.

- [Flux.1-fill-dev](https://huggingface.co/black-forest-labs/flux.1-fill-dev).
- [ICEdit-normal-LoRA](https://huggingface.co/RiverZ/normal-lora/tree/main).

Note: Due to some cooperation permission issues, we have to withdraw the weights and codes of moe-lora temporarily. What is released currently is just the ordinary lora, but it still has powerful performance. If you urgently need the moe lora weights of the original text, please email the author.

## Inference in bash (w/o VLM Inference-time Scaling)

Now you can have a try!

> Our model can **only edit images with a width of 512 pixels** (there is no restriction on the height). If you pass in an image with a width other than 512 pixels, the model will automatically resize it to 512 pixels.

> If you found the model failed to generate the expected results, please try to change the `--seed` parameter. Inference-time Scaling with VLM can help much to improve the results.

```bash
python scripts/inference.py --image assets/girl.png \
                            --instruction "Make her hair dark green and her clothes checked." \
                            --seed 304897401 \
```

Editing a 512×768 image requires 35 GB of GPU memory. If you need to run on a system with 24 GB of GPU memory (for example, an NVIDIA RTX3090), you can add the `--enable-model-cpu-offload` parameter.

```bash
python scripts/inference.py --image assets/girl.png \
                            --instruction "Make her hair dark green and her clothes checked." \
                            --enable-model-cpu-offload
```

If you have downloaded the pretrained weights locally, please pass the parameters during inference, as in: 

```bash
python scripts/inference.py --image assets/girl.png \
                            --instruction "Make her hair dark green and her clothes checked." \
                            --flux-path /path/to/flux.1-fill-dev \
                            --lora-path /path/to/ICEdit-normal-LoRA
```

## Inference in Gradio Demo

We provide a gradio demo for you to edit images in a more user-friendly way. You can run the following command to start the demo.

```bash
python scripts/gradio_demo.py --port 7860
```

Like the inference script, if you want to run the demo on a system with 24 GB of GPU memory, you can add the `--enable-model-cpu-offload` parameter. And if you have downloaded the pretrained weights locally, please pass the parameters during inference, as in:

```bash
python scripts/gradio_demo.py --port 7860 \
                              --flux-path /path/to/flux.1-fill-dev (optional) \
                              --lora-path /path/to/ICEdit-normal-LoRA (optional) \
                              --enable-model-cpu-offload (optional) \
```

Then you can open the link in your browser to edit images.

<div align="center">
<div style="width: 80%; text-align: left; margin:auto;">
    <img style="width:100%" src="docs/images/gradio.png">
    <p style="text-align: left;">Gradio Demo: just input the instruction and wait for the result!</b>.</p>
</div>

<div align="left">

Here is also a Chinese tutorial [Youtube video](https://www.youtube.com/watch?v=rRMc5DE4qMo) on how to install and use ICEdit, created by [softicelee2](https://github.com/softicelee2). It's definitely worth a watch!

## ComfyUI Workflow

### ComfyUI-nunchaku

We extend our heartfelt thanks to @[judian17](https://github.com/judian17) for crafting a ComfyUI [workflow](https://github.com/River-Zhang/ICEdit/issues/1#issuecomment-2846568411) that facilitates seamless usage of our model. Explore this excellent [workflow](https://github.com/River-Zhang/ICEdit/issues/1#issuecomment-2846568411) to effortlessly run our model within ComfyUI. 

The checkpoint utilized in this workflow differs slightly from the one described in the original paper. Please redownload it from [here](https://huggingface.co/RiverZ/normal-lora/tree/main).  This workflow incorporates high-definition refinement, yielding remarkably good results. Moreover, integrating this LoRA with Redux enables outfit changes to a certain degree. Once again, a huge thank you to @[judian17](https://github.com/judian17) for his innovative contributions! For more details about the workflow, please refer to this [issue](https://github.com/River-Zhang/ICEdit/issues/1#issuecomment-2846568411).

![comfyui image](docs/images/comfyuiexample.png)


#### TODO: Workflow for the MoE ckpt

### 🎨 Enjoy your editing! 



# Comparison with Commercial Models

<div align="center">
<div style="width: 80%; text-align: left; margin:auto;">
    <img style="width:100%" src="docs/images/gpt4o_comparison.png">
    <p style="text-align: left;">Compared with commercial models such as Gemini and GPT-4o, our methods are comparable to and even superior to these commercial models in terms of character ID preservation and instruction following. <b>We are more open-source than them, with lower costs, faster speed (it takes about 9 seconds to process one image), and powerful performance</b>.</p>
</div>


<div align="left">


# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=River-Zhang/ICEdit&type=Date)](https://www.star-history.com/#River-Zhang/ICEdit&Date)

# Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@misc{zhang2025ICEdit,
      title={In-Context Edit: Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer}, 
      author={Zechuan Zhang and Ji Xie and Yu Lu and Zongxin Yang and Yi Yang},
      year={2025},
      eprint={2504.20690},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.20690}, 
}
```
