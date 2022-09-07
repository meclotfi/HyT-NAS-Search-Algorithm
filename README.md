# HyT-NAS-Search-Algorithm
 <img src="https://github.com/meclotfi/HyT-NAS-Search-Algorithm/blob/main/assets/motivation.png" width="50%" align="right"/>
This project contains the search strategy used by our method HyT-NAS. HyT-NAS automates the search for tiny and edge hybrid attention and convolution architectures.
This search strategy is an optimized multi-objective bayesian optimization (MOBO) For fast convergence. 

We have used in this repos the [DGEMO](https://github.com/yunshengtian/DGEMO) implementation of MOBO method and added multiple component to it: 
- New surrogate models (Neural Networks, XgBoost, Bayasian Neural Networks ,ect)
- New selection strategies (Non dominated sorting ,etc).
- New sampling methods (Beta, Clustering ,etc).

The promise of HyT-NAS is to offer a simple yet efficient strategy to take hybrid convolution and attention -based networks from the cloud realm up to the tiny limit. 

## Run the method 
You can run the method on our search space following the notebook run_method.ipynb, you can also adapt the method to any search space by adapting the Problem Folder according to your respresenations and method of evaluation. you can also change the components of the strtegy using the algorithm.py file

## How it works
Our strategy allows fast convergence with minimum evaluations, especially on huge search spaces.
The Figure below shows the pipeline of our search algorithm:

![alt text](https://github.com/meclotfi/HyT-NAS-Search-Algorithm/blob/main/assets/SA.png?raw=true)

• First, an initial population of architectures A0 = {a1, ..., an} is sampled via latent hyperbolic sampling [].
the elements of the population will then be evaluated to construct a dataset D = (a1, y1), ..., (an, yn) where
yi is a tuple containing the accuracy and latency of the architecture ai. the dataset will grow incrementally
through a sequence of iterations of our method.

• In each iteration, the surrogate model is trained using the dataset containing all previously evaluated points. the
predictions of the model are then used to approximate the objectives via an acquisition function.

• Next, the surrogate problem defined as minimizing the acquisition function is solved using a multi-objective
solver. the optimal points found by the solver will go through a selection process based on the value of Hyper-
Volume Improvement (HVI). this selection will result in the construction of the new population Ai that will be
evaluated and added to the dataset D.

# Additional Results

<img src="https://github.com/meclotfi/HyT-NAS-Search-Algorithm/blob/main/assets/detect.png" width="50%" align="right"/>
Our study includes object detection models. These models usually involves a backbone, a neck and a head. **The backbone** extracts multi-resolution features from the input image. **The neck** will then aggregates these features. **The head** generates the final predictions. 
We analyzed the execution time of various models on a Raspberry Pi to know where the focus on optimization is required. 

The figure on the right shows how the backbone represents the main overhead for the latency of the models.
