# HyT-NAS-Search-Algorithm
 
This repos contain the search strategy used by our method HyT-NAS to search for tiny and edge hybrid attention and convolution architectures,
This search strategy is an optimized multi-objective bayesian optimization (MOBO) For fast convergence. 

We have used in this repos the DGEMO implementation of MOBO method and added multiple component to it like: 
- New surrogate models (Neural Networks, XgBoost, Bayasian Neural Networks ,ect)
- New selection strategies (Non dominated sorting ,etc).
- New sampling methods (Beta, Clustering ,etc).


Evaluating the accuracy and latency
of each sampled architecture is the bottleneck component of
HW-NAS. Hence, our strategy allows fast convergence with
minimum evaluations, especially on huge search spaces.
Figure 3 shows the pipeline of our search algorithm:
• First, an initial population of architectures A0 =
{a1, ..., an} is sampled via latent hyperbolic sampling [].
the elements of the population will then be evaluated
to construct a dataset D = (a1, y1), ..., (an, yn) where
yi is a tuple containing the accuracy and latency of
the architecture ai. the dataset will grow incrementally
through a sequence of iterations of our method.
• In each iteration, the surrogate model is trained using
the dataset containing all previously evaluated points. the
predictions of the model are then used to approximate the
objectives via an acquisition function.
• Next, the surrogate problem defined as minimizing the
acquisition function is solved using a multi-objective
solver. the optimal points found by the solver will go
through a selection process based on the value of Hyper-
Volume Improvement (HVI). this selection will result in
the construction of the new population Ai that will be
evaluated and added to the dataset D.
Surrogate model: In the standard pipeline, MOBO methods
use the Gaussian process (GP) as a surrogate to model
each objective independently. GPs are known for their low
performance on high dimensional data (d>10) and their poor
scalability with respect to the number of evaluations, which
make them unsuitable for our problem. Thus, we replace them
with a set of rank predictors of different depths and widths.
Rank predictors are models that tries to rank the values of the
target instead of approximating it. These models are trained
at each iteration on all previously evaluated architectures (D).
then, the average and the standard deviation of the network’s
predictions are used to approximate the rank of the objectives
via an acquisition function.
Acquisition function: a function that assigns a score to each
probable observation based on its likelihood of assisting the
optimization objective. Points that are judged promising by
this function are most likely to be selected by the multi-
objective solver to be part of the next population. The most
common acquisition functions used in literature include ex-
pected improvement (EI), probability of improvement (PI),
and upper confidence bound (UCB). The latter is the one we
choose in our search strategy.
Multi-objective solver: the solver is used to find solutions
to the surrogate problem defined as optimizing the acquisition
function. For that we use the evolutionary algorithm NSGA2.
the output of the solver is the population produced by the last
generation of NSGA2. This output is then transferred to the
selection strategy.
Selection strategy: Our selection strategy involves two steps:
First, points obtained from the solver are sorted using non-
dominated sorting algorithm, Then, the points belongs to the
two first dominance levels are passed to the second selection
phase based on hyper-volume improvement where the next
population is constructed iteratively by selecting the point with
the biggest hypervolume improvement each time.
