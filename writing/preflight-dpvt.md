# Deep neural networks for Phylogenetic inference Via Traversals (DPVT)

## The "pitch"

Likelihood and parsimony phylogenetic inference work because they consider data on a tree.
Optimization algorithms don't stare at the sequence alignment and then, separately, look at a tree and try to understand how to improve it.
Rather, they do tree traversal and think about the sequences in the context of the tree.
This then enables incremental improvement of the tree.

The goal of this project is to leverage these strategies for phylogenetic inference using deep neural networks.


## What are you trying to do? Articulate your objectives using words that would be familiar to someone who had taken an undergraduate class in the topic.

Develop phylogenetic inference methods that can learn from experience, beyond simply learning "hyperparameters" for optimization routines.

More specifically, I would like to propose methods that 

1. guide incremental improvement on a tree rather than spitting out a complete tree
2. use tree traversal to integrate data in a phylogenetically relevant way
3. use alignment columns to compare to other relevant parts of the alignment

As a first step, I propose that we work in a parsimony framework.
Given an input tree and represent the sequence data on an input tree using, for example, a run of the Fitch algorithm.
Modulo ambiguous sites, this is a 1-1 representation of a sequence alignment that provides a phylogenetically relevant structure on which we can think about mutations.
Furthermore, I propose that the output of this network is a classification of edges of the input tree as being "good" or "bad", i.e. in or not in the "correct" tree (details at bottom).
This will greatly reduce the space on which the phylogenetic algorithm is operating.

For inference, I propose that we have something analogous to a [DAG-NN](https://openreview.net/forum?id=JbuYF437WB6) that does neural network computation via traversals.
The DAG to traverse will be the input tree (though there's nothing keeping us from using an hDAG if a tree seems like it's working).

Here are a couple of proposed differences:

1. Our output is not graph-wise, but rather edge-wise on the input tree, classifying if an edge is in the true tree or not
2. We do computation sitewise, i.e. per alignment column, each of which uses the same NN. So the per-node features are a concatenation of the traversals across sites.
3. We could consider doing full recurrent calculation rather than local calculation that passes through multiple layers.
4. They have a specific aggregation operator, which is basically a weighted average using attention, and we could do something else:
    * We could have a full dense neural network that takes two inputs and returns an output, then also some NN to modify the representation given a mutation along an edge
    * We could actually skip this whole NN traversal part, and have the NN look at Fitch sets or some other artifact from a classical traversal

We then need to think about a way to aggregate the per-site-per-node information into a per-edge prediction.
**Note** that this will be a little tricky because different sequence alignments will have different numbers of informative sites, but there are techniques to deal with that.


## Who cares? If you're successful, what difference will it make?

Lots of phylogenetic algorithms are run on similar data sets, and having a method that incorporates prior experience could be very useful.


## How is it done today, and what are the limits of current practice?

The leading edge for this sort of work today are the current crop of deep neural networks.
They don't provide competitive performance for anything other than very small data sets.

This proposal is related to the work of [Azouri ... Pupko](https://www.nature.com/articles/s41467-021-22073-8) which given a tree and a proposed SPR move, uses summary statistics of that tree-move pair to predict its eventual likelihood.
Both are ML methods that aim to improve tree search.
However:

* Here we wish to classify in an absolute sense if an edge is correct or not, which is not something that their method can do
* We use sequence features directly

It also feels like there are some connections between this and [the aLRT test](https://paperpile.com/shared/FFu6Fr) for predicting branch support based on local likelihoods.
Both this and that use the sequence alignment and the tree to make a prediction.
However:

* The goal here isn't to predict uncertainty, it's to find ways to improve the tree
* I think an assumption of the aLRT test is that you start with a fully optimized tree

There is existing literature on branch and bound.
I don't know much about it, but it doesn't seem to be used in practice.


## What's new in your approach and why do you think it will be successful?

I've long felt like NNs seem too interesting to pass up for phylogenetic inference, however:

1. The multiclass problem on a superexponential space simply isn't going to work, no matter how clever the architecture and big the data.
2. I don't love treating the data as just a one-hot encoded sequence alignment. It just doesn't feel like an easy thing to hand to an algorithm because the algorithm has to figure out a way to reckon the mutations in an appropriate phylogenetic context.


## What is a best case scenario hypothetical result? Try to be as specific as possible. Use your imagination!

To have a method that can recognize suboptimal regions of a tree.
These could then be optimized using standard moves.
A next step would be for the algorithm to suggest these moves.


## What are the potential bad outcomes? Any overall concerns here?

1. Will the algorithm learn anything?
2. Will it be limited to the simplest phylogenetic models? Given the limited overlap between parsimony optimality and BEAST output, will this be useful for anything?
3. Will it be too slow to be useful? Likelihood computation is, in contrast, simple linear algebra.
    * Response: We will only need to do a single forward-backward pass on the tree to get prediction, not lots.
    * Response-response: What about backprop?
4. Perhaps parsimony and likelihood algorithms are so fast at evaluating SPRs that any attempt to make them "smart" is misguided.
5. Will we be limited to parsimony improvement?
6. One can imagine a bad situation in which the algorithm has a "blind spot" that takes us to a local optimum and we never escape because we aren't trying big SPRs.


## Is there pre-existing work/code that could be leveraged to explore the potential for bad outcomes? To do proof-of-concept investigation to get a first-pass answer for the underlying scientific question?

I'd like to think more about this.


## Are there any other categorically different approaches that could be applied here?

None that I can think of other than existing things like phyloformer.


## If this is a methods project, what methods will you compare to? Can you get them running before writing new code?

Benchmark speed and results against classical heuristic search.


## What data will you use? Are there appropriate hold-out sets?

Simulated data and real data, see below.


## Is it possible that better data would make this project irrelevant?

No.


## Sketch the approach, broken down into steps, with expected amounts of time and intermediate steps for each.

Overall approach is to train on pairs $(X_i, Y_i)$, where 

* $X_i$ is a suboptimal tree onto which we have mapped mutations (e.g. using Fitch) for a corresponding alignment
* $Y_i$ is a vector indicating which of those edges are in the "correct tree"

What does "correct" mean?
Correct could mean that it was the tree used for simulation if we are doing simulated data, or it is a very fully optimized tree if we are using real data.
I like real data, and so I vote for the latter.
The objective then becomes: let's come up with a fast way of finding ways to improve the tree.
This is something we can then evaluate at test time.

Harry's point: let's make sure the perturbed tree is suboptimal (we could do this by calculating a parsimony or likelihood score), which is fast.

Also, note that given a correct tree and a sequence alignment, according to any definition, we can create lots of examples of perturbed trees and corresponding $Y_i$s.


<!--
### Stage 1 (X months):



#### How do we decide to move onto the next stage?
-->

