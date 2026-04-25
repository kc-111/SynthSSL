1. Isotropic gaussian (small ones) for the views? Hinge loss on invariance term to avoid views collapsing to the center and being exactly the same? Whenever we use augmentations to get more robust representations, we may need this mechanism. We should consider making a shape and invariance experiment to demonstrate this exactly and when it fails.
2. For video or dynamic world model, how about koopman or residual based loss instead of discrete time prediction loss?
3. I think data is what enforces semantic structure. For still things like images, technically another trivial solution would be mapping images and their augmentations to some point in space without regard to semantic structure and the dataset latent distribution is isotropic gaussian. This means that we need predictive losses. For example, I hypothesize semantic structure can be created via actions, such as we have two actions: left/right/top/down. We have an artificial order of image classes, the actions would then cause association of the latent structure and organize it in that way. Left/Right encodes completely different classes and top/down are different types of whatever is in there. Compared to simply still image joint embedding, will this increase semantic structure? The experiment will maybe be forcing 5 image classes into 2D and looking at the 2D space. For higher dimensions, we may need a similarity evaluation metric.

For 1:
How to design the self supervised training for this?

For 3:
Is semantic structure from co occurence (similar objects appearing)?