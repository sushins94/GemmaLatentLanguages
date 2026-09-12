# GemmaLatentLanguages
Exploring Latent Languages in Gemma3 models 


NOTE: 
finite_difference_check in run_all incorrectly reports, cosine below 0.3 means the Jacobian is wrong. 
This is actually fine.

(Claude Opus5 comment on JLens fitting)
Low cosine at early layers is expected behaviour, not a failure.
This is the documented weakness of J-lens and the reason R-lens
exists (Blank, Bhatia & Nanda 2026, which replaces the raw gradient with
layer-wise relevance propagation precisely to reduce this accumulation). 

J_l is an average over 25 prompts x ~123 source positions so the true Jacobian varies
a lot across contexts in early layers, so the corpus average describes no
single prompt well there. And the check compares against the true directional
derivative on one prompt, which is the quantity an average cannot match.

Observed on gemma-3-4b-pt fitted on corpora/en.txt:

  layer  1: cos +0.019 +/- 0.127   norm ratio 0.208   
  layer 11: cos +0.082 +/- 0.173   norm ratio 0.392   
  layer 22: cos +0.375 +/- 0.088   norm ratio 0.645   
  layer 31: cos +0.853 +/- 0.015   norm ratio 0.971   


The rise toward the target layer is the signature to look for. Layer 31 sits
one block below the target (layer 32 of 34), so J_31 should be near identity,
and cos 0.853 with norm ratio 0.971 confirms the fitting code is correct. The
hard pass/fail test is that J at the target layer IS the identity, true by
construction regardless of model or corpus; the finite-difference cosine is a
fidelity measure, not a correctness test.

CONSEQUENCE FOR THE RESULTS: fidelity is depth-dependent, so J-lens readouts
are reliable late, marginal mid-stack (~0.4 around layer 22, which is where the
English peak sits in the class-0 reproduction), and unreliable early. Any
J-lens curve should be read with that in mind, and cross-lens comparisons
should use RANKS rather than probabilities, which are less sensitive to the
transport's noise and scale.