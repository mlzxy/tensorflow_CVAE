# Conditional Variational Auto Encoder
> 变分编码器

This is a tensorflow VAE implementation on MNIST data set, containing three networks:

1. Basic [VAE](VAE): generate digit images from random noise, could also use to remove noise or occlusion removal
2. Conditional VAE to generate digits [CVAE](CVAE_from_digits): generate digit images from certain label.
3. Conditional VAE to remove occlusion [CVAE_occlusion](CVAE_occlusion): has special decoder part to remove Image occlusion. 

## Demo

### Basic

For example, random digits generated: 

![](VAE/results/output_images/random_generated/0.generated_random.png)  
![](VAE/results/output_images/random_generated/1.generated_random.png)  
![](VAE/results/output_images/random_generated/3.generated_random.png)  
![](VAE/results/output_images/random_generated/4.generated_random.png)  
![](VAE/results/output_images/random_generated/51.generated_random.png)
![](VAE/results/output_images/random_generated/33.generated_random.png)  
![](VAE/results/output_images/random_generated/24.generated_random.png)  
![](VAE/results/output_images/random_generated/54.generated_random.png) 

### CVAE From Digits

Following is generated images for number 3:

![](CVAE_from_digits/results/generated_images/0.generated.label-3.png)
![](CVAE_from_digits/results/generated_images/1.generated.label-3.png)
![](CVAE_from_digits/results/generated_images/3.generated.label-3.png)
![](CVAE_from_digits/results/generated_images/4.generated.label-3.png)
![](CVAE_from_digits/results/generated_images/55.generated.label-3.png)
![](CVAE_from_digits/results/generated_images/33.generated.label-3.png)
![](CVAE_from_digits/results/generated_images/99.generated.label-3.png)
![](CVAE_from_digits/results/generated_images/12.generated.label-3.png)
### CVAE Noise/Occlusion removal

For example:

original: ![](CVAE_occlusion/results/output_images/1.origin.png)
occluded: ![](CVAE_occlusion/results/output_images/1.occluded.png)
infered: ![](CVAE_occlusion/results/output_images/1.reconstructed.png) (has inferred shape in the occluded area)

But it generally works not so good, using Basic VAE could even perform better, I need to experiment more on the network. 


## How to use

In each network dir, there is for example [VAE/results/save](VAE/results/save) dir. This is the saved tensorflow model, copy it to [VAE](VAE), and run `python test_generation.py`.  You will get generated images in [VAE/save](VAE/save). The other two networks are similarly used. 








