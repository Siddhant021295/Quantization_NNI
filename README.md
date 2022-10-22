# Quantization_NNI

### Requirements
- python 3.10.4
- torch
- torchvision
- nni
- time

### how to sunning the code

- python vae.py

### Results

#### Project Learnings

- As we can see that the inference time for the quantized model is reduced in each of the different quantization approaches we have taken. This is obtained since the compact model discretizes the weights and activation function used in the forward and backward pass, thus making multiplication operations faster, consequently reducing inference time.
- Another important factor to be considered while quantization is the test set loss for the model. We can generally see an increase in the average and test set loss for each model-configuration combination because the quantization of weights and activations produces a non-optimal result and thus a larger loss compared to the original model. Though, we obtain a better inference time and memory consumption which is a trade-off that can be considered when trying to run such models on smaller less-memory devices.
- We can also notice slight differences in the test set loss and inference time taken by the different types of quantizers that we have tried, namely Naive, QAT, and BNN. We can notice a high loss in BNN because it is discretizing our weights to +1 or -1 which is generating trivial results and thus resulting in a high loss. Although, we notice that BNN quantized models have the lowest inference time for the same reasons that it discretizes the weights, simplifying the multiplication operations to bitwise operations.
- Additionally, we cannot observe a decrease in model memory consumption since we do not perform the speed-up operation that calibrates the model and optimizes it for the given backend hardware. If we are able to perform speedup, we would obtain a reduced inference time along with memory efficiency.
