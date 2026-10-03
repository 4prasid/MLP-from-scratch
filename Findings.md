# Findings

This document is the full written analysis behind the project's findings; a durable, text-only companion to the [W&B report](https://wandb.ai/prasid-indian-institute-of-technology-madras/assignment_1/reports/DA6401-Assignment-1-PH21B007-PRASID--VmlldzoxNjEyODA5Ng?accessToken=7lf6abidol3880zy7aiflc38domgwf0gtrwlsqz0fhboc9dumm1bdqjfn0fs1042), which holds all the plots, parallel-coordinate visualizations, and gradient-norm charts referenced below.

---

## 1. Class Similarity & Visual Overlap

Five sample images per class were logged to a W&B table to visually inspect the dataset before training.

Several digit pairs show clear structural overlap:
- **4 and 9** - when written loosely, both can look nearly identical: one typically has a steep curve, the other a smooth one, but handwriting variation blurs this distinction.
- **3 and 5** - share curved horizontal strokes that overlap significantly in pixel space.
- **2 and 8** - the curved bottom of a 2 can resemble the loops of an 8.

**Why this matters for an MLP specifically:** these visual similarities create regions in the 784-dimensional flattened-pixel feature space where different classes have very similar representations. Because the network operates directly on raw pixels with no spatial feature extraction (no convolutions), it cannot exploit structural differences the way a CNN could. As a result, ambiguous handwriting styles that fall into these overlapping regions cause the model to make **confident but wrong** predictions - a pattern that shows up directly in the Error Analysis section below.

---

## 2. Hyperparameter Sweep

A 100-run Bayesian-optimization sweep was performed, maximizing `val_f1_score`, over the following search space:

```python
'epochs':        {'values': [10, 15, 20]},
'batch_size':    {'values': [32, 64, 128]},
'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 0.1},
'optimizer':     {'values': ['sgd', 'momentum', 'nag', 'rmsprop']},
'activation':    {'values': ['relu', 'sigmoid', 'tanh']},
'weight_init':   {'values': ['random', 'xavier']},
'num_layers':    {'values': [2, 3, 4, 5]},
'hidden_size':   {'values': [16, 32, 64, 128]},
'loss':          {'values': ['cross_entropy', 'mse']},
'weight_decay':  {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1}
```

**Best result:** validation F1 = **0.97753**, with configuration:

```
epochs: 20, batch_size: 128, learning_rate: 0.001818, optimizer: rmsprop,
activation: relu, weight_init: xavier, num_layers: 2, hidden_size: 128,
loss: cross_entropy, weight_decay: 0.00050751
```

**What mattered most:** per the Parameter Importance and Parallel Coordinates plots, **optimizer** had the single largest impact on validation F1 - `rmsprop` runs consistently scored highest, with strong positive correlation. **Loss function** was the second most impactful factor: cross-entropy correlated positively, MSE negatively, confirming cross-entropy is the better fit for multi-class classification. **Learning rate** showed moderate importance but near-zero linear correlation - it's sensitive rather than monotonic: a good value (0.001–0.01) helps a lot, but both too-low and too-high values give similarly poor scores. **Sigmoid activation** correlated negatively, consistently underperforming ReLU and Tanh. Architectural knobs - hidden size, number of layers, batch size, weight initialization - all showed near-zero importance, indicating that **for MNIST with an MLP, optimizer and loss function choice matter far more than network size.**

(Note: a second sweep view using `val_accuracy` as the target metric ranked **loss type** as the single most important hyperparameter, with optimizer=RMSprop second; weight decay showed negative correlation, meaning excessive regularization hurt accuracy on MNIST. Both metrics point to the same conclusion - optimizer and loss function dominate architecture choices for this task.)

---

## 3. Optimizer Comparison

All four implemented optimizers (SGD, Momentum, NAG, RMSprop) were compared on an identical architecture: 3 hidden layers, 128 neurons each, ReLU activation, 10 epochs, batch size 64, Xavier init, learning rate 0.001818 (the best found rate), zero weight decay (to isolate optimizer behavior), cross-entropy loss.

**Result:** RMSprop minimized loss by far the fastest - reaching a training loss of **0.032** and **97.3% validation accuracy** by epoch 10, converging from the very first epoch. SGD, Momentum, and NAG all remained stuck above 2.0 training loss across all 10 epochs, making essentially no progress.

**Why, theoretically:** RMSprop maintains a per-parameter adaptive learning rate by dividing each gradient by the root-mean-square of its recent gradient history. Image classification inputs (flattened pixels) have highly variable gradient magnitudes - frequently-activated pixels produce large gradients, sparse/background pixels produce small ones. A fixed learning rate (as in plain SGD) applies the same step size to all of them, causing oscillation on large-gradient parameters and near-zero progress on small-gradient ones. RMSprop automatically scales down updates where gradients are large and scales up where they're small, giving faster, more stable convergence - especially well suited to high-dimensional, sparse inputs like flattened images. (Adam and Nadam, which combine this adaptive scaling with momentum, are expected to perform similarly well or better for the same underlying reason, though they were not included in this project's training runs - see the main README for details.)

---

## 4. Vanishing Gradients: Sigmoid v/s ReLU

RMSprop + Xavier init was held fixed while comparing ReLU vs. Sigmoid activation at 2 and 4 hidden layers (128 neurons/layer, batch size 128, lr=0.001818, zero weight decay, cross-entropy). The L1 gradient norm of the first hidden layer was logged across 10 epochs.

**Result:** the gradient norm plot shows vanishing gradients clearly with Sigmoid. ReLU maintained healthy gradient norms (0.004–0.008) at *both* depths throughout training. Sigmoid at 4 layers showed severely suppressed norms (0.00004–0.002), and even Sigmoid at 2 layers showed near-zero norms (~0.002) from the start - the gradient was essentially vanishing before reaching the first hidden layer.

The downstream effect: both ReLU configurations converged to ~97% validation accuracy within a few epochs. Sigmoid at 2 layers reached only ~92%. Sigmoid at 4 layers started near random chance (~10%) and only slowly climbed to ~86% by epoch 10 - a severely delayed convergence directly caused by the vanishing gradient.

**Why, theoretically:** the Sigmoid derivative has a maximum value of 0.25 (at z=0) and approaches 0 at both extremes. During backpropagation, the gradient is multiplied by this derivative at every layer. With 4 layers, the gradient can shrink by up to 0.25⁴ ≈ 0.004, making it nearly impossible for early layers to receive a meaningful learning signal. ReLU avoids this entirely - its derivative is either exactly 0 or exactly 1, so gradients pass through active neurons completely unchanged regardless of network depth.

---

## 5. Dead Neuron Investigation

A 3-hidden-layer network (128 neurons each, Xavier init, batch size 128, cross-entropy, zero weight decay) was trained with RMSprop at a deliberately high learning rate (0.1) using ReLU, and compared against an identical Tanh run and a normal-learning-rate ReLU run (lr=0.001818).

**Result:** ReLU at lr=0.1 accumulated 348 dead neurons by epoch 2 - roughly **90% of all 384 neurons** (3 × 128) became permanently inactive after just one epoch. Validation accuracy plateaued at ~10% (random chance) for all 10 epochs, with the gradient norm collapsing to near zero.

**Mechanism:** the large learning rate caused massive weight updates that pushed neuron pre-activations deeply negative. Once a ReLU neuron's pre-activation is negative for every input in the dataset, it outputs exactly zero permanently, and its gradient (`dL/dW = dL/dA * (Z > 0)`) becomes exactly zero too - making recovery mathematically impossible. This is the classic "dying ReLU" problem.

The normal-lr ReLU run (0.001818) showed **zero** dead neurons throughout and converged to ~97% accuracy with healthy gradient norms (0.004–0.007), confirming that learning rate - not the activation function itself - was the trigger.

The Tanh run at lr=0.1 showed zero dead neurons, as expected: tanh's derivative, `1 − tanh²(z)`, is always strictly positive, so dead neurons are structurally impossible. However, Tanh *also* failed to converge at this learning rate, plateauing at ~10% accuracy with near-zero gradient norms - but for a different reason: the high learning rate caused repeated weight overshooting, pushing neurons into tanh's saturated regions (output near +1 or −1, where gradients approach 0). This is gradient vanishing via **saturation**, not neuron death.

**Takeaway:** Tanh is immune to dead neurons but still vulnerable to saturation at high learning rates; ReLU's failure mode is more severe and irreversible, since dead neurons never recover.

---

## 6. Loss Function Comparison

Cross-entropy and MSE were compared on an identical architecture (3 hidden layers, 128 neurons, ReLU, Xavier init, RMSprop, lr=0.001818, zero weight decay, batch size 128, 10 epochs).

**Result:** Cross-entropy converged faster and to a higher final accuracy. From epoch 1, cross-entropy already reached ~92% validation accuracy vs. MSE's ~90%, and this gap persisted throughout training - cross-entropy reached ~97.3% by epoch 10 vs. MSE's ~96.5%. (Note: train/val loss values themselves aren't directly comparable between the two runs, since MSE and cross-entropy operate on different numerical scales - accuracy is the fair comparison metric here.)

**Why, theoretically, cross-entropy is better suited to multi-class classification with softmax:**
1. The gradient of cross-entropy with respect to the softmax input simplifies cleanly to `(predicted − true)` - a direct signal proportional to the prediction error, with no extra terms. MSE paired with softmax instead produces a gradient involving the full softmax Jacobian (`s * (dA − sum(dA * s))`), a weaker and noisier learning signal.
2. Cross-entropy directly maximizes the log-likelihood of the correct class, penalizing confident wrong predictions very heavily (since `log(near zero)` → a very large loss), producing strong corrective gradients. MSE treats all errors quadratically and doesn't strongly penalize confident wrong predictions, making it less effective at pushing the model toward sharp, confident, *correct* predictions in a classification setting.

---

## 7. Global Performance Analysis

Every one of the 100 sweep runs was plotted as training accuracy vs. validation accuracy, colored by learning rate.

**Result:** three broad clusters emerge.
- **Top-right (train > 0.90, val > 0.90):** the largest cluster, dominated by low learning rates (0.001–0.01). Points sit very close to the diagonal - minimal train/val gap, strong generalization. These correspond to RMSprop + cross-entropy runs that converged successfully.
- **Middle (train 0.6–0.8, val 0.65–0.83):** moderate performance with a small but visible gap, indicating mild underfitting - reasonable learning rates but suboptimal optimizer/loss choices that hadn't fully converged within 10 epochs.
- **Bottom-left (train ≈ 0.1, val ≈ 0.09):** complete underfitting - SGD, MSE, or extreme learning rates that failed to learn anything meaningful.

**Notably, no severe overfitting was observed** anywhere - even the best-performing points stay close to the diagonal (train_acc ≈ val_acc). Two reasons: (1) Bayesian optimization actively steers subsequent runs away from poorly-performing hyperparameter regions after observing early results, naturally reducing the number of extreme/overfit configurations explored; (2) MNIST is a reasonably large dataset (48,000 training samples) while model capacity is capped at 128 neurons per layer, making it hard for the network to memorize the training set outright. Any train/val gap observed in the middle cluster reflects runs still *converging*, not overfitting - suggesting more epochs, not more regularization, would help those specific runs.

---

## 8. Error Analysis

The best model (RMSprop, ReLU, 2 layers, 128 neurons, lr=0.001818, weight decay=0.00050751, Xavier init) was evaluated on the full MNIST test set (10,000 samples).

**Result:** 9,793 / 10,000 correctly classified. The most frequent confusions: **9 → 4** (12 cases), **5 → 3** (10 cases), **2 → 7** (8 cases) - consistent with the visual-similarity patterns identified earlier. Class 5 was the weakest performer (867/892 correct); class 1 was the strongest (1,127/1,135 correct), intuitively sensible since "1" is the most visually distinct digit.

**Most confident failures:** the 5 test samples the model was most confident about yet got wrong - all exceeding **99% confidence despite being incorrect**:
- A true "8" (written in an unusual cursive style resembling a 4) predicted as "4" at 99.82% confidence.
- A true "6," written very narrowly and vertically, predicted as "1" at 99.76% confidence.
- A true "2" predicted as "8" at 99.55% confidence, due to the curved bottom of the 2 resembling the loops of an 8.

**Interpretation:** these high-confidence failures show the model has learned strong - but sometimes misleading - visual features. It isn't simply *uncertain* on hard cases; it is genuinely and confidently misled by unusual handwriting styles that fall in overlapping regions of the learned feature space, exactly as anticipated in the Class Similarity analysis.

---

## 9. Weight Initialization & Symmetry Breaking

Two otherwise-identical training runs (3 hidden layers, 128 neurons, Sigmoid activation, RMSprop, lr=0.001818, zero weight decay, batch size 128, cross-entropy) were compared, differing only in weight initialization: **Zero** vs. **Xavier**. The gradient norms of 5 individual neurons within the same hidden layer were tracked over the first 50 iterations.

**Result:** with Zero initialization, all 5 neuron gradient-norm lines overlap **perfectly** throughout all 50 iterations, at values on the order of 10⁻¹³ - essentially identical and near-zero for every neuron at every step. With Xavier initialization, all 5 lines are clearly distinct from iteration 0, with magnitudes up to ~4×10⁻⁵, confirming each neuron learns independently from the start. Correspondingly, validation accuracy with Xavier reached **85% within 5 epochs**, while Zero initialization stayed stuck at ~10% (random chance) throughout.

**Why zero initialization fails - the symmetry problem:** with all weights set to zero, every neuron in a layer receives identical input weights, computes identical pre-activations, and produces identical outputs for every input. During backpropagation, since all neurons have identical forward activations, they receive identical error signals and identical gradient updates. The gradient norm for every neuron in the layer is mathematically forced to be identical - exactly what the perfectly overlapping Zero-init plot shows. Because every neuron updates by the same amount in the same direction at every step, the symmetry is **self-perpetuating forever**: a layer of 128 identically-initialized neurons is mathematically equivalent to a single neuron, completely eliminating the network's representational capacity regardless of width or depth. Even though the *total* loss can decrease slightly under zero init, all 128 neurons are learning the exact same single feature simultaneously - the network wastes 127 of 128 neurons per layer.

**Why Xavier works - symmetry breaking is a mathematical necessity:** Xavier samples each weight from a distribution with a specific variance, giving every neuron a different starting point. Different initial weights → different forward activations → different error signals → different gradients (exactly what the Xavier plot shows, with all 5 lines distinct from the very first iteration). This lets each neuron specialize into a different feature detector, giving the network its full representational power - directly confirmed by Xavier reaching 85% accuracy while Zero init fails to learn anything.

---

## 10. Fashion-MNIST Transfer

Given a constrained budget of only 3 hyperparameter configurations, and strictly using learnings from the MNIST experiments above (RMSprop outperforms other optimizers; cross-entropy beats MSE; Xavier init is essential; learning rates of 0.001–0.01 work well), three configurations were chosen for Fashion-MNIST:

1. **Config 1 - Best MNIST model directly reused:** RMSprop, lr=0.001818, ReLU, 2 layers × 128 neurons, batch size 128, weight decay 0.00050751, Xavier init → **~90% validation accuracy, ~0.90 F1** by epoch 20.
2. **Config 2 - Deeper network:** RMSprop, lr=0.001, ReLU, 4 layers × 128 neurons, batch size 64, weight decay 0.001, Xavier init → **~88–89% validation accuracy**. Depth and regularization were increased on the hypothesis that Fashion-MNIST's more complex visual patterns (clothing textures) would benefit from more capacity.
3. **Config 3 - Tanh activation:** RMSprop, lr=0.001, Tanh, 3 layers × 128 neurons, batch size 64, weight decay 0.001, Xavier init → **~88% validation accuracy**. Tanh's smooth, bounded gradients were hypothesized to help with complex clothing textures, but performed slightly worse than ReLU.

**Result:** Config 1 - the best MNIST configuration, transferred as-is - also performed best on Fashion-MNIST, confirming that the core learnings (RMSprop + cross-entropy + Xavier init + moderate learning rate) generalize well across datasets.

However, all three configurations showed more oscillation and slower, lower-ceiling convergence on Fashion-MNIST compared to MNIST - MNIST converged smoothly to 97%+, while Fashion-MNIST stabilized around 88–90% with visible fluctuations even at epoch 20. This reflects Fashion-MNIST's greater inherent complexity: clothing items (shirts, coats, pullovers) share many overlapping visual features - sleeves, collars, textures - unlike digits, which have more geometrically distinct shapes. A pure MLP operating on flattened pixels cannot capture the spatial hierarchies present in clothing patterns as effectively as it captures digit strokes, which explains both the lower ceiling accuracy and the need for more epochs to approach convergence.
