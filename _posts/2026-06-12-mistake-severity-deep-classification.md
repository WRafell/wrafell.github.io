---
title: "Mistake Severity in Deep Classification"
date: 2026-06-15
categories:
  - notes
tags:
  - deep-learning
  - image-classification
  - medical-ai
  - mistake-severity
excerpt: "Not all misclassifications are equally bad. A review of two papers that rethink the training objective and evaluation metric to account for the semantic — and clinical — severity of errors."
mathjax: true
---
# Mistake Severity in Deep Classification

## Introduction

Standard image classification pipelines are trained and evaluated under a fundamental but rarely questioned assumption: all misclassifications are equally bad. A model that predicts "wolf" when the ground truth is "dog" is penalized exactly as much as one that predicts "chair". Cross-entropy loss and top-$k$ accuracy are both blind to the *severity of an error*; e.g., the semantic distance between the predicted class and the true one.

This blindness is not merely a theoretical concern. In high-stakes domains such as medical diagnosis, an error that confuses a benign growth with an early-stage malignancy is qualitatively different from one that confuses it with a clearly unrelated condition. A model that is **equally wrong** in both senses may be **catastrophically wrong** in practice.

The two papers discussed in this review directly address this gap from complementary perspectives. *Making Better Mistakes* (Bertinetto et al., 2020) revisits the neglected problem of hierarchy-aware classification and proposes symmetric mistake-severity metrics and losses grounded in class taxonomies. *Every Error has Its Magnitude* (Hong et al., 2026) takes this into the medical domain, introducing an asymmetric formulation that acknowledges the directional nature of clinical error severity, embedded within a Multiple Instance Learning (MIL) framework for whole slide image (WSI) analysis.

**Motivation.** These papers illustrate a six-year arc of an underexplored problem. Bertinetto et al. showed in 2020 that the deep learning community had stopped caring about mistake quality; Hong et al. showed in 2026 that the problem persists in medical AI, where error asymmetry has real clinical consequences. They argue that going beyond accuracy requires rethinking both the training objective and the evaluation metric.

This connects to a broader critique in the AI community. For instance, Lun Wang (2025) argues that **if the evaluation metric is wrong, optimizing it produces models that look good on paper but fail in deployment**. The two papers reviewed here are a concrete instance of exactly that failure, not in some hypothetical future system, but classifiers being trained and evaluated today.

**Disclaimer.** This article is not focused on the empirical results of the reviewed papers (please refer to the papers), but on the respective core ideas and methodologies.

---

## Priors

**Class Hierarchies and Taxonomies.** Many classification tasks are naturally structured into hierarchies. In ImageNet, classes are organized according to WordNet, a large lexical database where nouns are arranged in hypernym/hyponym relationships. For example, "German Shepherd" is a hyponym of "dog," which is itself a hyponym of "mammal." In medical imaging, diagnostic categories are organized by clinical severity or anatomical specificity. These hierarchies encode meaningful semantic distance between classes, which flat classification ignores.

**Multiple Instance Learning (MIL).** MIL is a weakly supervised paradigm particularly suited to computational pathology. A gigapixel whole slide image (WSI) is treated as a *bag* of thousands of patches (*instances*), with only a bag-level label available during training. A MIL model learns to aggregate instance-level representations into a bag-level prediction, typically using attention-based pooling (Ilse et al., 2018). MIL is the dominant framework for WSI-level diagnosis tasks such as cancer subtyping and grading.

**Standard Cross-Entropy Loss.** For a $C$-class problem with input $x$, parameters $\mathbf{w}$, and ground truth label $y$, the standard cross-entropy loss is:

$$\ell_{\text{CE}}(x, y; \mathbf{w}) = -\log \frac{\exp(f_y(x; \mathbf{w}))}{\sum_{c=1}^{C} \exp(f_c(x; \mathbf{w}))}$$

This treats all classes $c \neq y$ as equally wrong, assigning no weight to whether a mistake is semantically close (e.g., predicting "cat" instead of "dog") or absurd (e.g., predicting "airplane" instead of "dog").

---

## Making Better Mistakes: Leveraging Class Hierarchies with Deep Networks

Bertinetto et al. (2020) reframes image classification as a problem where not all errors are equal, and proposes two simple modifications of the cross-entropy loss to reduce the *severity* of mistakes by leveraging the class hierarchy. The paper's central observation is striking: while top-1 error improved dramatically over this period, the *severity* of the mistakes that were made remained nearly unchanged. The community had been optimizing the wrong objective.

**Problem Setting.** The hierarchy $\mathcal{H}$ is a tree over the label set $\mathcal{C}$. The classifier outputs a categorical distribution $p(C) = \phi_C(x; \theta)$ over leaf classes. The goal is to learn $\theta$ such that, when mistakes occur, they are semantically close to the true class in $\mathcal{H}$. The paper evaluates on two modified datasets: **tieredImageNet-H** (608 classes, tree of height 13, derived from WordNet) and **iNaturalist-H** (1010 classes, 8-level complete tree, biological taxonomy). Both are modified from their originals to ensure the hierarchy is a well-formed tree compatible with the loss functions.

### Methodology

The paper proposes two loss functions, each a one-parameter generalization of the standard cross-entropy that reduces to it in the appropriate limit.

#### Hierarchical Cross-Entropy (HXE)

When $\mathcal{H}$ is a tree, the class probability $p(C)$ factorizes along the path from leaf $C$ to the root $\mathcal{R}$:

$$p(C) = \prod_{l=0}^{h-1} p\!\left(C^{(l)} \mid C^{(l+1)}\right)$$

where $C^{(0)} = C$, $C^{(h)} = \mathcal{R}$, and each conditional is computed from the leaf probabilities as $p(C^{(l)} \mid C^{(l+1)}) = \sum_{A \in \text{Leaves}(C^{(l)})} p(A) \;/\; \sum_{B \in \text{Leaves}(C^{(l+1)})} p(B)$. HXE is then the reweighted sum of the cross-entropies of these conditionals:

$$\mathcal{L}_{\text{HXE}}(p, C) = -\sum_{l=0}^{h-1} \lambda(C^{(l)}) \log p\!\left(C^{(l)} \mid C^{(l+1)}\right)$$

The weight assigned to each edge is $\lambda(C^{(l)}) = \exp(-\alpha \cdot h(C^{(l)}))$, where $h(C^{(l)})$ is the height of node $C^{(l)}$ and $\alpha > 0$ is a hyperparameter. Nodes *closer to the root* (higher $h$) receive *lower* weight, meaning the loss discounts coarse-level distinctions and focuses on fine-grained ones. When $\alpha \to 0$, all weights equal 1 and $\mathcal{L}_{\text{HXE}}$ reduces to the standard cross-entropy.

#### Soft Labels

The second approach replaces the one-hot target with a soft distribution over all classes, where probability mass decays exponentially with hierarchical distance from the true class. For ground truth $C$ and any class $A$, the soft label is:

$$y^{\text{soft}}_A(C) = \frac{\exp(-\beta \cdot d(A, C))}{\sum_{B \in \mathcal{C}} \exp(-\beta \cdot d(B, C))}$$

where $d(C_i, C_j) = h(\text{LCA}(C_i, C_j)) / h_{\text{tree}}$ is the *normalized* LCA height (divided by the total tree height), and $\beta > 0$ controls the concentration of the distribution. Here, LCA means Lowest Common Ancestor — the deepest node in the hierarchy tree that is a shared ancestor of two classes. Large $\beta$ recovers the one-hot case; small $\beta$ approaches a uniform distribution. Standard cross-entropy is then applied using $y^{\text{soft}}$ as the target. This method has a natural interpretation as encoding the labeler's uncertainty that arises from visual confusion between closely related classes.

#### Hierarchy-Aware Evaluation Metrics

The paper uses two metrics based on the height of the LCA between the predicted and true class.

**Hierarchical distance of a mistake.** The LCA height $h(\text{LCA}(\hat{y}, y))$ is computed only for *misclassified* examples (where $\hat{y} \neq y$) and averaged over all such examples. This measures the severity of errors when a single top-1 prediction is considered.

**Average hierarchical distance of top-$k$.** For each test example, the mean LCA height between the ground truth $y$ and each of the $k$ most likely predictions is computed, then averaged over the test set. This is relevant when multiple hypotheses can be considered by a downstream system, and is reported for $k \in \{1, 5, 20\}$.

### Key Findings and Limitations

The paper demonstrates an inherent **tradeoff between top-1 accuracy and mistake severity**: as $\alpha$ increases (HXE) or $\beta$ decreases (soft labels), the hierarchical distance of mistakes improves, but top-1 error increases. Both methods offer Pareto improvements over prior art: for any desired tradeoff point, HXE or soft labels achieve lower hierarchical error than DeViSE, YOLO-v2, or Barz & Denzler at the same top-1 accuracy. At the low-severity extreme, HXE with $\alpha = 0.1$ matches cross-entropy top-1 accuracy while reducing the hierarchical distance of mistakes on both datasets.

The key limitation is that both the loss and the metric treat errors *symmetrically*: the cost of predicting a coarser class than the truth is treated identically to the cost of predicting a class in the wrong branch entirely. There is no notion of directionality, which is clinically essential. A second limitation, noted in the hidden stratification literature, is that aggregate metrics like the average hierarchical distance can mask failures concentrated in specific subgroups even when the overall score improves.

---

## Every Error has Its Magnitude: Asymmetric Mistake Severity Training

Hong et al. (2026) addresses a critical limitation of the previous hierarchy-aware methods, including Bertinetto et al. (2020): they treat the severity of a mistake symmetrically. In clinical diagnosis, however, misclassifying an urgent finding as a trivial one is incomparably more severe than the reverse. This paper introduces **PAMS** (**P**riority-**A**ware **M**istake **S**everity), an MIL-based training framework designed to address the asymmetry of misclassification risk in multiclass WSI diagnosis.

**Problem Setting.** Each WSI $X_a$ is treated as a bag of patch-level instances $\{x_{a,k}\}_{k=1}^{n(X_a)}$. Although individual patch labels $y_{a,k}$ are unknown during training, the bag-level label $Y_a$ is assigned as the most severe diagnosis present among all patches $Y_a = \max_c\!\left(\{y_{a,k}\}_{k=1}^{n(X_a)}\right)$, where classes are ordered by clinical urgency: $0 \prec 1 \prec \cdots \prec C-1$. A pre-trained feature extractor maps patches to encoded instances, which a MIL aggregator combines to predict the bag label $\hat{Y}_a$. To leverage the diagnostic hierarchy, PAMS trains one MIL classifier $f_{\theta_h}$ per hierarchy level $h$, from the finest level $\mathcal{H}$ (e.g., 7 cancer subtypes) up to a single root $\mathcal{R}$ (e.g., Benign / Atypical / Malignant), each with a separate classification head over the shared MIL backbone.

### Methodology

#### Mistake Severity Cross-Entropy (MSCE)

Standard cross-entropy treats all misclassifications within a hierarchy level equally. To account for directional urgency, the authors define a regularization weight matrix $M^h = [M^h_{ij}] \in \mathbb{R}^{C_h \times C_h}$ at hierarchy level $h$, where $\alpha > 1$ is a scale hyperparameter:

$$M^h_{ij} = \begin{cases} \alpha^{|i-j|}, & \text{if } c^h_i \succ c^h_j \\ 1, & \text{otherwise} \end{cases}$$

This matrix applies a stronger penalty when the model predicts a lower-urgency class ($c^h_j$) while the truth is a higher-urgency class ($c^h_i \succ c^h_j$), but not the reverse. The MSCE loss is then:

$$\mathcal{L}_{\text{MSCE}} = -\sum_{h=1}^{H} \underbrace{\hat{p}^h M^h (\tilde{Y}^h)^\top}_{\text{directional weight}} \sum_{c=0}^{C_h - 1} \tilde{Y}^h[c] \log \hat{p}^h[c]$$

where $\hat{p}^h$ is the predicted probability vector and $\tilde{Y}^h$ is the one-hot target at level $h$. The scalar $\hat{p}^h M^h (\tilde{Y}^h)^\top$ serves as a directional regularization weight: it is large when the model assigns high probability to a lower-urgency class while the truth is urgent, and small otherwise.

#### Hierarchy Alignment Loss

Since each hierarchy classifier $f_{\theta_h}$ operates on the same WSI, their predictions should be consistent across levels. The authors enforce this via a Jensen-Shannon Divergence (JS) between predictions at consecutive hierarchy levels:

$$\mathcal{L}_{\text{HA}} = \sum_{h=1}^{H-1} \text{JS}(\hat{p}^h \| \dot{p}^{h+1})$$

where $\dot{p}^{h+1}$ is the predicted probability at level $h+1$ marginalized to level $h$ by summing over child classes. The full training objective is:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{MSCE}} + \lambda_2 \mathcal{L}_{\text{HA}}$$

#### Semantic Feature Remix (SFR)

MIL systems must learn to prioritize the most urgent finding when multiple pathologies co-exist in a WSI. However, such complex-finding samples are rare and difficult to annotate. SFR generates synthetic complex-finding bags by mixing instances from two WSIs with different priority labels $Y_a \succ Y_b$.

Given bags $Z_a$ and $Z_b$, all instances are clustered into $L$ clusters. Clusters are sorted by the proportion of instances originating from $Z_a$: the highest-ranked clusters are dominated by the unique findings of the higher-priority bag $X_a$. The top-$k$ clusters from $Z_a$ are then added to $Z_b$ to form a synthesized bag $Z_{a+b}$, which inherits the higher-priority label $Y_a$. This selection ensures that the most diagnostically relevant patches from the urgent case are included in the synthetic sample.

#### Asymmetric Mikel's Wheel Severity Metric

To evaluate mistake severity, the authors propose two metrics based on an **asymmetric Mikel's Wheel**. For a prediction $\hat{Y}^h = c^h_i$ and true label $Y^h = c^h_j$, the confusion weight is:

$$W^h_{i,j} = 1 + |i - j| + \mathbb{1}(c^h_i \succ c^h_j) \times P, \quad \forall W^h_{i,j} \geq 1$$

where $P \geq 1$ is an additional penalty applied only when the model under-predicts urgency (i.e., predicts a less urgent class than the truth). The matrix $W$ is asymmetric: $W^h_{i,j} \neq W^h_{j,i}$. Two metrics are derived from $W$ and the confusion matrix $S^h$ (where $S^h_{i,j}$ is the number of samples from class $c^h_i$ predicted as $c^h_j$):

$$\text{AsCC} = \frac{1}{\sum_{i,j} S^h_{i,j}} \sum_{i=0}^{C_h-1} \sum_{j=0}^{C_h-1} S^h_{i,j} \times \frac{1}{W^h_{i,j}}$$

$$\text{AsMC} = \frac{1}{\sum_{i \neq j} S^h_{i,j}} \sum_{i=0}^{C_h-1} \sum_{j=0}^{C_h-1} \mathbb{1}(i \neq j) \times \frac{S^h_{i,j}}{W^h_{i,j} - 1}$$

**AsCC** (Asymmetric Classification Confidence) measures mistake severity across *all* WSI-level predictions using the asymmetric weight $W^h_{i,j}$. A perfect model scores 1; dangerous under-predictions (missing urgent diagnoses) reduce the score more than safe over-predictions. **AsMC** (Asymmetric Misclassification Confidence) restricts the same calculation to *misclassified* WSIs only, directly quantifying how dangerous the errors are when they occur. Both metrics operate at the bag level: since patch labels are never available, they are computed from the WSI-level confusion matrix $S^h$ aggregated over the test set. Higher is better for both, as both are inverse-penalty scores: they divide by $W^h_{i,j}$ rather than multiplying, so a model that errs in the safe direction scores higher than one that misses urgent diagnoses.

### Limitations

The urgency ordering and penalty $P$ in PAMS are fixed at training time based on clinical judgment, with no mechanism to detect or adapt when grading schemes evolve. Additionally, since patch-level labels are unknown, the severity signal operates only at the bag level; i.e., if a misclassification originates from a minority of patches, it may not propagate effectively to the most clinically relevant instances.

On the augmentation side, SFR's cluster-based patch selection assumes the feature extractor produces well-separated representations for adjacent severity classes, which is hardest to guarantee precisely at the boundary cases the method targets. More broadly, both $M^h$ and Mikel's Wheel assume monotonically ordered severity, limiting applicability to clinical tasks with non-ordinal or branching diagnostic hierarchies.

---

## Discussion

Both **Making Better Mistakes** and **Every Error has Its Magnitude** respond to the same root problem: *optimizing for accuracy treats all misclassifications as equivalent, even when some errors are far more consequential than others.* They outline a coherent research program, moving from symmetric hierarchy-aware classification in natural images to asymmetric, clinically grounded severity training in medical AI.

**The metric shapes what gets optimized.** A deeper issue raised by both works, and relevant to how we think about AI evaluation broadly, is that **the choice of metric shapes what gets optimized**. When a community standardizes on top-1 accuracy, models are trained to maximize it, and improvements in other dimensions go unmeasured and unrewarded. This is an instance of Goodhart's Law: *once a measure becomes a target, it ceases to be a good measure*. The six-year gap between these two papers suggests that the computer vision community largely set aside the mistake severity problem after the deep learning revolution made flat accuracy easy to improve. Hong et al. suggest this complacency has real costs in medical AI.

**Dynamic hierarchies.** Both papers assume a fixed, manually specified hierarchy. Diagnostic classifications evolve (i.e., grading schemes are revised, new subtypes are discovered), and a severity-aware model trained on today's hierarchy may be silently miscalibrated to tomorrow's clinical standards.

**Geometric embeddings as a path forward.** Rather than fixing the hierarchy as a discrete graph, it could be encoded in a continuous, learnable space. **Order embeddings** represent entailment via coordinate-wise ordering; **box embeddings** use hyperrectangle containment and intersection volume as a differentiable replacement for LCA height; **hyperbolic embeddings** embed trees in a Poincaré ball with exponentially expanding volume, well-suited to deep diagnostic taxonomies. These approaches could both replace the manually specified hierarchy and provide a data-driven foundation for the asymmetric weights $w^-$ and $w^+$ in Hong et al.

**Utility-based evaluation.** Graph distance is a structural proxy for what we actually care about: patient outcomes. Decision theory offers a more principled alternative: a utility function $U(y, \hat{y})$ encoding the expected cost of action $\hat{y}$ when the truth is $y$. Proper scoring rules such as the Brier score are special cases of this framework and are *incentive-compatible*: a model cannot improve its score without improving its predictions. Neither LCA height nor Mikel's Wheel has this property.

**Calibration.** Neither paper evaluates calibration, a.k.a., whether predicted confidence scores are reliable. Overconfident wrong predictions on high-severity cases are precisely the failure mode clinicians are least equipped to catch. Severity-weighted training may worsen this by pushing stronger commitments to high-severity predictions. Evaluating calibration per severity level, rather than in aggregate, would be a natural and important extension.

**Human-AI team performance.** Both papers evaluate the model in isolation, as if it acts autonomously. In clinical practice, the model recommends, and a pathologist decides. **Human-AI complementarity** is the appropriate target: does the team produce fewer high-severity errors than either alone? This is not guaranteed. Overreliance (clinicians accepting confident AI recommendations without scrutiny) is most dangerous precisely on high-severity cases, where the model is most likely to be confidently wrong and the clinician most likely to defer. A model that reduces mistake severity in solo evaluation could paradoxically increase it in deployment.

**Conclusion.** The arc from Bertinetto et al. to Hong et al. represents genuine progress: from ignoring mistake severity entirely, to encoding it symmetrically, to modeling it asymmetrically in a clinically grounded setting. Yet each step has also revealed a deeper layer of the same problem. Fixing the loss without fixing the metric is insufficient. Fixing both without grounding them in utility is insufficient. And evaluating the model without evaluating the human-AI system it will become part of is insufficient. The core lesson is that **what we measure determines what we build**, and building AI that is genuinely safe requires closing the gap between the metrics we optimize and the outcomes we actually care about.

---

## Appendix: Background Concepts

**WordNet.** WordNet is a large lexical database for English in which nouns, verbs, adjectives, and adverbs are grouped into sets of cognitive synonyms (synsets), each expressing a distinct concept. Synsets are interlinked by conceptual-semantic and lexical relations. For ImageNet, the class hierarchy is derived from the WordNet noun hierarchy, providing a principled graph structure over the 1000 (or more) leaf classes.

**Lowest Common Ancestor (LCA).** Given a directed graph $\mathcal{H}$ (the class hierarchy) and two nodes $a$, $b$, the Lowest Common Ancestor $\text{LCA}(a, b)$ is the deepest node that is an ancestor of both $a$ and $b$. The height $h(\text{LCA}(a, b))$ is commonly used as a measure of semantic distance: a shallow LCA means the two classes are semantically distant (e.g., "dog" and "airplane"), while a deep LCA means they are closely related (e.g., "German Shepherd" and "Labrador Retriever").

**Attention-Based MIL.** The canonical attention-based MIL model computes a bag-level representation as a weighted sum of instance features:

$$\mathbf{z} = \sum_{k=1}^{K} a_k \mathbf{h}_k, \quad a_k = \frac{\exp(\mathbf{w}^\top \tanh(\mathbf{V}\mathbf{h}_k))}{\sum_{j=1}^{K} \exp(\mathbf{w}^\top \tanh(\mathbf{V}\mathbf{h}_j))}$$

where $\mathbf{h}_k$ is the feature vector of the $k$-th instance, and $a_k$ is a scalar attention weight learned by a small neural network with parameters $\mathbf{w}$ and $\mathbf{V}$. The bag representation $\mathbf{z}$ is then passed to a classifier head.

**Jensen-Shannon Divergence (JS).** The Jensen-Shannon Divergence is a symmetric, bounded measure of the similarity between two probability distributions $P$ and $Q$. It is defined as:

$$\text{JS}(P \| Q) = \frac{1}{2} \text{KL}(P \| M) + \frac{1}{2} \text{KL}(Q \| M), \quad M = \frac{P + Q}{2}$$

where $\text{KL}(P \| M) = \sum_x P(x) \log \frac{P(x)}{M(x)}$ is the Kullback-Leibler divergence from $M$ to $P$. Unlike KL divergence, JS is symmetric ($\text{JS}(P\|Q) = \text{JS}(Q\|P)$) and always finite, with values in $[0, \log 2]$.

In the context of PAMS, JS is used in the hierarchy alignment loss $\mathcal{L}_{\text{HA}}$ to enforce consistency between the predictions of two consecutive hierarchy classifiers $f_{\theta_h}$ and $f_{\theta_{h+1}}$. Intuitively, if the model predicts "high-grade malignancy" at the fine-grained level, its coarser-level prediction should also reflect high urgency. Using JS rather than KL ensures the penalty is symmetric: it does not matter which hierarchy level is treated as the reference.

---

## Appendix: Evaluation Metrics Summary

Both papers propose metrics that go beyond top-$k$ accuracy to capture mistake quality.

| Metric (Paper) | Definition | Direction |
|---|---|---|
| Top-$k$ error *(Both)* | Fraction of examples where the ground truth is not among the top-$k$ predictions. Treats all mistakes equally. | Lower $\downarrow$ |
| Hier. dist. of a mistake *(Bertinetto)* | Mean $h(\text{LCA}(\hat{y}, y))$ over misclassified examples only ($\hat{y} \neq y$). | Lower $\downarrow$ |
| Avg. hier. dist. top-$k$ *(Bertinetto)* | Mean $h(\text{LCA}(\hat{y}_i, y))$ over the $k$ most likely predictions per example, averaged over the test set. $k \in \{1,5,20\}$. | Lower $\downarrow$ |
| AsCC *(Hong et al.)* | $\frac{1}{\sum_{i,j} S^h_{i,j}} \sum_{i,j} \frac{S^h_{i,j}}{W^h_{i,j}}$, with $W^h_{i,j} = 1 + \lvert i-j \rvert + \mathbb{1}(c^h_i \succ c^h_j) P$. Severity over all samples. | Higher $\uparrow$ |
| AsMC *(Hong et al.)* | $\frac{1}{\sum_{i \neq j} S^h_{i,j}} \sum_{i \neq j} \frac{S^h_{i,j}}{W^h_{i,j}-1}$. Same $W^h_{i,j}$, restricted to misclassified samples only. | Higher $\uparrow$ |

**Key distinction between the two families.** The LCA-based metrics of Bertinetto et al. are *symmetric*: $h(\text{LCA}(\hat{y}, y)) = h(\text{LCA}(y, \hat{y}))$, so over-predicting and under-predicting urgency are penalized equally. The AsCC and AsMC metrics of Hong et al. are *asymmetric*: the weight $W^h_{i,j}$ applies an additional penalty $P$ when the model under-predicts urgency ($c^h_i \succ c^h_j$, i.e., the true class is more urgent than the predicted one), reflecting the clinical reality that missing a high-grade diagnosis is more dangerous than over-calling it.

---

**Sources:** Bertinetto et al., 2020 — *Making Better Mistakes: Leveraging Class Hierarchies with Deep Networks*; Hong et al., 2026 — *Every Error has Its Magnitude: Asymmetric Mistake Severity Training*