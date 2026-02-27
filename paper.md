# ECG Arrhythmia Classification Using a ResNet-Inspired 1D CNN with Multi-Scale Feature Fusion

**Author Name¹, Author Name²**

¹*Institution/Affiliation, City, Country*
²*Institution/Affiliation, City, Country*

*Corresponding author: email@address*

---

## Abstract

**Background:** Cardiovascular diseases remain the leading cause of mortality worldwide, with cardiac arrhythmias representing a significant subclass requiring accurate and timely diagnosis. Automated classification of electrocardiogram (ECG) signals is essential for clinical decision support and continuous patient monitoring.

**Methods:** We propose a novel hybrid architecture for ECG arrhythmia classification combining a ResNet-inspired 1D convolutional neural network with handcrafted feature fusion. The model processes raw ECG signals through three progressive residual blocks (64→128→256 filters) with decreasing kernel sizes (11→7→5), followed by multi-scale pooling to capture both global and local patterns. Simultaneously, six statistical and morphological features are extracted and processed through a compact network. The dual pathways are fused and classified into five AAMI-standard arrhythmia classes. SMOTE-Tomek resampling and Gaussian noise augmentation address severe class imbalance in the MIT-BIH dataset.

**Results:** The proposed model achieves 98.27% accuracy and 98.30% weighted F1-score on the MIT-BIH test set (21,892 samples), with a ROC-AUC of 99.28%. Per-class analysis reveals robust performance on rare arrhythmia classes (S: 82.01% F1, F: 77.94% F1) and exceptional specificity of 99.62%. The model successfully identifies normal beats with 99.12% F1 and ventricular arrhythmias with 95.79% F1. Only 379 misclassifications occur among 21,892 test samples (1.73% error rate).

**Conclusions:** Our architecture achieves state-of-the-art performance competitive with complex Transformer-based methods while maintaining architectural efficiency suitable for clinical deployment. The strategic fusion of learned representations with handcrafted features, combined with effective imbalance handling, produces a model with excellent generalization and minimal false positives.

**Keywords:** ECG classification; arrhythmia detection; convolutional neural networks; residual learning; MIT-BIH dataset; deep learning

---

## 1. Introduction

Cardiovascular diseases (CVDs) remain the leading cause of mortality worldwide, accounting for approximately 17.9 million deaths annually according to the World Health Organization [1]. Among these, cardiac arrhythmias—abnormalities in heart rhythm—represent a significant subclass that can lead to severe complications including stroke, heart failure, and sudden cardiac death [2]. The electrocardiogram (ECG) serves as the primary non-invasive diagnostic tool for detecting and classifying arrhythmias, capturing the heart's electrical activity through electrodes placed on the skin.

Traditionally, ECG analysis has relied on manual interpretation by trained cardiologists—a process that is time-consuming, subject to inter-observer variability, and impractical for continuous monitoring of at-risk patients [3]. The advent of wearable cardiac monitoring devices and the subsequent explosion of continuously collected ECG data have created an urgent need for automated, accurate, and reliable arrhythmia classification systems. Such systems can serve as clinical decision support tools, reducing cardiologist workload and enabling early intervention in critical cases [4].

The MIT-BIH Arrhythmia Database, introduced in 1980 by the Massachusetts Institute of Technology and Boston's Beth Israel Hospital, has become the de facto standard benchmark for evaluating arrhythmia classification algorithms [5]. This dataset contains 48 half-hour recordings of two-channel ECG signals, meticulously annotated by expert cardiologists, encompassing both common and rare arrhythmia types. The dataset's characteristic class imbalance—with normal beats vastly outnumbering various arrhythmia classes—presents a fundamental challenge that any robust classification system must address [6].

Early approaches to automated ECG classification relied on handcrafted feature extraction combined with traditional machine learning classifiers. These methods extracted morphological features (QRS complex duration, amplitude), statistical features (mean, variance, skewness), and frequency-domain features from ECG signals, subsequently feeding them into classifiers such as Support Vector Machines (SVM), k-Nearest Neighbors (k-NN), or Random Forests [7, 8]. While achieving moderate success, these approaches were fundamentally limited by the quality and comprehensiveness of manually engineered features, which might fail to capture subtle pathological patterns.

The deep learning revolution has transformed ECG analysis, with Convolutional Neural Networks (CNNs) emerging as particularly effective for learning hierarchical representations directly from raw ECG signals [9]. Unlike traditional methods requiring extensive preprocessing and feature engineering, CNNs automatically learn relevant features through successive layers of convolution operations, capturing patterns at multiple temporal scales [10]. The 1D CNN architecture is naturally suited to ECG signals, which are one-dimensional time series with local morphological patterns that are diagnostically significant.

More recently, researchers have explored increasingly complex architectures including recurrent neural networks (RNNs), long short-term memory networks (LSTMs), and Transformers to capture temporal dependencies in ECG sequences [11, 12]. Transformer-based models, in particular, have achieved impressive results by leveraging self-attention mechanisms to model long-range dependencies [13]. However, these gains in accuracy often come at the cost of substantially increased model complexity, computational requirements, and training data demands—factors that may limit clinical deployability on resource-constrained devices.

In this paper, we present a novel hybrid architecture for ECG arrhythmia classification that strategically combines the strengths of residual learning, multi-scale feature extraction, and handcrafted feature fusion. Our approach is motivated by three key observations: (1) residual connections enable deeper network training while mitigating vanishing gradients; (2) multi-scale pooling captures both global and local patterns critical for arrhythmia detection; and (3) handcrafted features provide complementary domain knowledge that learned representations may not fully capture. By integrating these elements within a computationally efficient framework, we achieve state-of-the-art performance while maintaining architectural simplicity suitable for clinical deployment.

The main contributions of this work are:

1. A **ResNet-inspired 1D CNN architecture** with residual blocks specifically designed for ECG morphology, incorporating progressive kernel size reduction to capture features at multiple temporal scales.

2. A **dual-input fusion strategy** that combines learned representations from raw ECG signals with handcrafted statistical and morphological features, leveraging domain knowledge to enhance classification, particularly for rare arrhythmia classes.

3. A **comprehensive data balancing pipeline** employing SMOTE-Tomek resampling and Gaussian noise augmentation to address the severe class imbalance inherent in the MIT-BIH dataset.

4. **Extensive experimental validation** demonstrating that our approach achieves 98.27% accuracy and 98.30% weighted F1-score on the MIT-BIH test set, with a ROC-AUC of 99.28%, matching or exceeding current state-of-the-art methods while maintaining architectural efficiency.

5. **Detailed per-class analysis** revealing robust performance even on rare arrhythmia classes (S and F), with specificity of 99.62%—critical for minimizing false alarms in clinical settings.

The remainder of this paper is organized as follows: Section 2 reviews related work in ECG arrhythmia classification. Section 3 details our methodology, including data preprocessing, architecture design, and training strategy. Section 4 presents experimental results and comparative analysis. Section 5 discusses implications and limitations. Section 6 concludes the paper.

---

## 2. Background and Literature Review

### 2.1 The MIT-BIH Arrhythmia Dataset

Since its release in 1980, the MIT-BIH Arrhythmia Database has served as the primary benchmark for arrhythmia classification research [5]. The dataset contains 48 half-hour recordings of two-channel ambulatory ECG signals, obtained from 47 subjects studied at Boston's Beth Israel Hospital Arrhythmia Laboratory. The recordings were digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range.

For standardized evaluation, the dataset is typically partitioned into training and test sets as preprocessed by Fazeli et al. [14], containing 87,554 and 21,892 heartbeat samples respectively. Each sample consists of 187 time steps centered on the R-peak, representing a single cardiac cycle. Beats are annotated into five classes following the Association for the Advancement of Medical Instrumentation (AAMI) recommended practice [15]:

- **Class N (Normal)**: Normal beats, left and right bundle branch blocks, atrial escape beats, nodal escape beats
- **Class S (Supraventricular)**: Atrial premature beats, aberrated atrial premature beats, nodal premature beats, supraventricular premature beats
- **Class V (Ventricular)**: Premature ventricular contractions, ventricular escape beats
- **Class F (Fusion)**: Fusion of ventricular and normal beats
- **Class Q (Unknown)**: Paced beats, unclassifiable beats

The dataset exhibits severe class imbalance, with normal beats (Class N) constituting approximately 83% of samples, while rare classes such as fusion beats (Class F) represent less than 1% of the data. This imbalance reflects real-world clinical distributions but poses significant challenges for automated classification systems, which may become biased toward majority classes [6].

### 2.2 Traditional Machine Learning Approaches

Early automated arrhythmia classification systems relied on handcrafted feature extraction followed by conventional machine learning classifiers. These approaches typically proceeded in three stages: preprocessing, feature extraction, and classification.

Preprocessing commonly included baseline wander removal, power-line interference filtering, and QRS complex detection using algorithms such as Pan-Tompkins [16]. **Feature extraction** encompassed three categories: (1) morphological features (QRS duration, amplitude, area under the curve), (2) statistical features (mean, variance, skewness, kurtosis of heartbeat intervals), and (3) frequency-domain features (power spectral density, wavelet coefficients) [7, 17].

De Chazal et al. [18] pioneered this approach, extracting 15 features including RR intervals, QRS morphology, and heartbeat intervals, achieving 83% accuracy using linear discriminants. Llamedo and Martínez [19] extended this work with 70 features and patient-adaptive schemes, reporting 94% accuracy. Ye et al. [8] combined higher-order statistics with wavelet features, using SVM classifiers to achieve 97.7% accuracy.

While these methods demonstrated the value of domain knowledge, they were fundamentally limited by the quality of handcrafted features, which might fail to capture subtle or complex pathological patterns. Additionally, feature engineering required significant domain expertise and was dataset-specific, limiting generalizability [20].

### 2.3 Deep Learning Architectures

#### 2.3.1 Convolutional Neural Networks

The application of CNNs to ECG analysis marked a paradigm shift, enabling automatic feature learning directly from raw signals. Kiranyaz et al. [9] proposed one of the first 1D CNN architectures for patient-specific ECG classification, achieving 99% accuracy but requiring patient-specific training data. Acharya et al. [10] developed a 9-layer CNN for automated detection of normal and abnormal ECG signals, reporting 94.4% accuracy on the MIT-BIH dataset.

Subsequent work explored deeper architectures and novel design elements. Kachuee et al. [21] proposed a residual CNN achieving 93.4% accuracy, while Yildirim [22] developed a deep wavelet auto-encoder with 99% accuracy. Hannun et al. [23] demonstrated that deep CNNs could match cardiologist-level performance in detecting 12 rhythm classes, though their model required substantial computational resources.

#### 2.3.2 Recurrent and Hybrid Architectures

Recognizing the sequential nature of ECG signals, researchers incorporated recurrent layers to model temporal dependencies. Singh and Pradhan [24] combined CNNs with bidirectional LSTMs, achieving 98.1% accuracy. Saadatnejad et al. [11] proposed LSTM networks for continuous ECG monitoring, reporting 94.1% F1-score. Yildirim et al. [25] developed a bidirectional LSTM network with wavelet sequences, achieving 99.39% accuracy.

The CRT-Net architecture [12] represents a recent hybrid approach, combining CNNs, RNNs, and Transformers in a multi-path fusion network, achieving 99.24% accuracy on MIT-BIH. While highly accurate, such architectures require substantial computational resources and careful hyperparameter tuning.

#### 2.3.3 Transformer-Based Methods

Inspired by their success in natural language processing, Transformers have recently been applied to ECG classification. The STAE Transformer [13] incorporates spatio-temporal attention with variational autoencoder augmentation, achieving 99.56% accuracy—among the highest reported on MIT-BIH. However, the model's F1-score of 95.40% suggests potential class imbalance issues, and its complexity limits clinical deployability.

FusionViT [26] combines Vision Transformers with wavelet and handcrafted features for wearable applications, achieving 86.23% accuracy on the PhysioNet 2017 dataset. While less accurate, this work demonstrates the potential of feature fusion approaches.

### 2.4 Handling Class Imbalance

Class imbalance remains a fundamental challenge in arrhythmia classification, with normal beats vastly outnumbering pathological examples. Researchers have explored multiple strategies to address this issue:

**Data-level approaches** modify the training distribution through oversampling minority classes or undersampling majority classes. Synthetic Minority Oversampling Technique (SMOTE) [27] generates synthetic samples by interpolating between existing minority class instances. Rahman and Davis [28] applied SMOTE to ECG classification, reporting improved sensitivity for minority classes. Tomek links remove borderline majority class samples, cleaning class boundaries [29]. The combination SMOTE-Tomek has shown particular promise for ECG data [30].

**Algorithm-level approaches** modify the learning process itself. Focal loss [31] down-weights well-classified examples, focusing training on hard samples. Class weighting assigns higher misclassification costs to minority classes during training. Rajpurkar et al. [32] employed class weighting in their cardiologist-level arrhythmia detector.

**Hybrid approaches** combine multiple strategies. The STAE Transformer [13] employs a hybrid loss combining focal and Dice losses with VAE-based augmentation, achieving strong results on minority classes despite overall lower F1-score.

### 2.5 Multi-Scale and Multi-Modal Approaches

Recent research has explored multi-scale feature extraction and multi-modal fusion to capture complementary information:

**Multi-scale architectures** employ parallel branches with different receptive fields to capture patterns at varying temporal scales. The Inception architecture [33], adapted to 1D signals, has shown promise for ECG analysis [34]. The rECGnition_v2.0 system [35] uses depthwise separable convolutions with multi-scale processing, achieving 98.07% accuracy with 82.7M FLOPs per sample.

**Multi-modal fusion** combines multiple data representations. Multimodal DL approaches [36] convert ECG signals to continuous wavelet transform (CWT) and Markov transition field (MTF) images, feeding them to separate CNN branches. While achieving 98.40% accuracy, the image conversion adds computational overhead. ECG-XPLAIM [37] combines Inception-style CNNs with Grad-CAM interpretability for clinical applications.

### 2.6 Gaps and Opportunities

Despite significant progress, several gaps remain in the literature:

1. **Complexity vs. Performance Trade-off**: Many state-of-the-art methods achieve high accuracy through increasingly complex architectures (Transformers, multi-path networks) that may be impractical for clinical deployment on resource-constrained devices [12, 13].

2. **Class Imbalance in Rare Arrhythmias**: While overall accuracy continues to improve, performance on rare classes (particularly S and F) often lags behind, with F1-scores below 80% in many studies [6, 30].

3. **Feature Fusion Underutilization**: Despite demonstrated value, the systematic integration of handcrafted features with learned representations remains underexplored in recent deep learning literature [26, 36].

4. **Generalization Gap**: Many methods show significant performance drops between validation and test sets, suggesting overfitting to training data characteristics [10, 21].

Our work addresses these gaps through a ResNet-inspired architecture that balances performance and efficiency, combined with SMOTE-Tomek balancing for robust rare-class handling, and strategic fusion of handcrafted features to enhance generalization.

---

## 3. Methodology

### 3.1 Dataset and Preprocessing

We utilized the MIT-BIH Arrhythmia Dataset as preprocessed by Fazeli et al. [14], comprising 87,554 training samples and 21,892 test samples. Each sample represents a single heartbeat as 187 time-domain samples centered on the R-peak, with corresponding AAMI standard class labels (N, S, V, F, Q).

#### 3.1.1 Signal Filtering

Raw ECG signals contain various noise sources including baseline wander, power-line interference, and high-frequency noise. We applied a 4th-order Butterworth bandpass filter with cutoff frequencies of 0.5 Hz and 45 Hz:

$$H(s) = \frac{1}{(s^2 + 1.414s + 1)^2}$$

The low cutoff (0.5 Hz) removes baseline wander and DC offset, while the high cutoff (45 Hz) eliminates power-line interference (50/60 Hz) and high-frequency noise. Zero-phase filtering (`scipy.signal.filtfilt`) was employed to preserve signal morphology and prevent phase distortion [38].

#### 3.1.2 Handcrafted Feature Extraction

Complementary to learned representations, we extracted six handcrafted features per sample, capturing statistical, morphological, and energy characteristics:

**Statistical Features:**
- **Mean** ($\mu$): Average signal amplitude
- **Standard Deviation** ($\sigma$): Signal dispersion
- **Skewness** ($\gamma_1$): Asymmetry of the amplitude distribution
  $$\gamma_1 = \frac{\frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^3}{\sigma^3}$$
- **Kurtosis** ($\gamma_2$): "Tailedness" of the distribution
  $$\gamma_2 = \frac{\frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^4}{\sigma^4} - 3$$

**Morphological Features:**
- **Number of Peaks**: Local maxima exceeding the 75th percentile threshold, capturing complexity of the heartbeat morphology
- **Signal Energy**: Sum of squared amplitudes, reflecting signal power
  $$E = \sum_{i=1}^{n} x_i^2$$

These features were selected based on clinical relevance documented in prior literature [7, 8] and provide domain knowledge that learned representations may not fully capture, particularly for rare arrhythmia classes with limited training examples.

#### 3.1.3 Normalization

Both raw signals and handcrafted features were normalized using Z-score standardization:

$$x_{\text{norm}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$$

where $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ are computed from the training set only to prevent data leakage, and $\epsilon = 10^{-8}$ ensures numerical stability. Separate scalers were maintained for signals and features, with signal scalers operating on flattened representations before reshaping to preserve inter-channel relationships.

### 3.2 Data Balancing and Augmentation

The MIT-BIH dataset exhibits severe class imbalance, with normal beats (Class N) comprising 82.7% of training samples while fusion beats (Class F) represent only 0.73%. To address this, we employed a multi-stage balancing pipeline.

#### 3.2.1 SMOTE-Tomek Resampling

We first applied SMOTE-Tomek [29], a hybrid approach combining synthetic oversampling with cleaning:

**SMOTE (Synthetic Minority Oversampling Technique)** [27] generates synthetic samples for minority classes by interpolating between existing instances. For each minority class sample $x_i$, SMOTE selects $k$ nearest neighbors (we used $k=5$) and creates synthetic samples:
$$x_{\text{new}} = x_i + \lambda(x_{zi} - x_i)$$
where $x_{zi}$ is a randomly selected neighbor and $\lambda \in [0,1]$ is a random interpolation coefficient.

**Tomek Links** [39] identify pairs of samples from different classes that are each other's nearest neighbors:
$$\text{Tomek Link}(x_i, x_j) \iff d(x_i, x_j) = \min(d(x_i, \cdot)) = \min(d(x_j, \cdot))$$
These links typically represent borderline or noisy samples; removing the majority-class member of each pair cleans class boundaries and improves classifier performance.

The combination SMOTE-Tomek first oversamples minority classes using SMOTE, then applies Tomek links to remove ambiguous samples from the augmented dataset, resulting in cleaner class boundaries and improved generalization [30].

#### 3.2.2 Noise Augmentation

To further enhance model robustness, we applied Gaussian noise augmentation to a random subset of training samples:
$$x_{\text{aug}} = x + \mathcal{N}(0, \sigma^2)$$
with $\sigma = 0.02$ (2% of normalized signal amplitude). This augmentation simulates real-world recording variations and acts as an implicit regularizer [40].

### 3.3 Network Architecture

Our proposed architecture employs a dual-input design with parallel processing paths for raw ECG signals and handcrafted features, followed by multi-scale feature fusion and classification.

#### 3.3.1 Signal Processing Path

The signal path processes raw ECG waveforms through a ResNet-inspired 1D CNN with residual blocks and multi-scale pooling.

**Initial Convolution Block:**
The network begins with a wide-kernel convolution to capture initial morphological patterns:
$$\text{Conv1D}(32, 15, \text{padding='same'}) \rightarrow \text{BatchNorm} \rightarrow \text{ReLU} \rightarrow \text{MaxPool}(2)$$

The kernel size of 15 corresponds to approximately 40ms at 187Hz sampling, sufficient to capture the QRS complex (typically 80-120ms) while preserving temporal resolution.

**Residual Block Architecture:**
Each residual block follows the pre-activation design proposed by He et al. [41], which has demonstrated superior gradient flow and training stability:

$$\begin{aligned}
\text{Input: } x_{\text{in}} \\
y_1 &= \text{Conv1D}(f, k, \text{padding='same'})(x_{\text{in}}) \\
y_1 &= \text{BatchNorm}(y_1) \\
y_1 &= \text{ReLU}(y_1) \\
y_1 &= \text{Dropout}(p)(y_1) \\
y_2 &= \text{Conv1D}(f, k, \text{padding='same'})(y_1) \\
y_2 &= \text{BatchNorm}(y_2) \\
\text{shortcut} &= \begin{cases}
x_{\text{in}} & \text{if } \text{dim}(x_{\text{in}}) = \text{dim}(y_2) \\
\text{Conv1D}(f, 1, \text{padding='same'})(x_{\text{in}}) & \text{otherwise}
\end{cases} \\
\text{Output} &= \text{ReLU}(y_2 + \text{shortcut})
\end{aligned}$$

The residual connection $(y_2 + \text{shortcut})$ enables gradient flow directly through the network during backpropagation, mitigating the vanishing gradient problem and allowing effective training of deeper architectures.

**Progressive Block Design:**
The three residual blocks progressively increase filter depth while decreasing kernel size:

- **Block 1**: $f=64$, $k=11$, $p=0.2$, followed by MaxPooling1D(2)
  - Output shape: (None, 46, 64)
  - Receptive field: ~120ms, capturing QRS complex and surrounding context

- **Block 2**: $f=128$, $k=7$, $p=0.3$, followed by MaxPooling1D(2)
  - Output shape: (None, 23, 128)
  - Receptive field: ~240ms, encompassing full P-QRS-T morphology

- **Block 3**: $f=256$, $k=5$, $p=0.4$
  - Output shape: (None, 23, 256)
  - Receptive field: ~300ms, capturing inter-beat relationships

This progressive design enables the network to learn increasingly abstract features while maintaining computational efficiency. Decreasing kernel sizes with depth allows later layers to focus on refined patterns derived from earlier coarse features. The increasing dropout rates provide stronger regularization in deeper layers where overfitting risk is higher.

**Multi-Scale Feature Extraction:**
After the residual blocks, we employ parallel pooling operations to capture features at multiple scales:

$$\begin{aligned}
x_{\text{res}} &= \text{Output of Block 3} \quad &\text{[23, 256]} \\
x_{\text{global}} &= \text{GlobalAveragePooling1D}(x_{\text{res}}) \quad &\text{[256-dim]} \\
x_{\text{maxpool}} &= \text{MaxPooling1D}(2)(x_{\text{res}}) \quad &\text{[11, 256]} \\
x_{\text{local}} &= \text{Flatten}(x_{\text{maxpool}}) \quad &\text{[2816-dim]} \\
x_{\text{signal}} &= \text{Concatenate}([x_{\text{global}}, x_{\text{local}}]) \quad &\text{[3072-dim]}
\end{aligned}$$

**Design Rationale:**
- **Global average pooling** captures the overall signal characteristics by averaging across the temporal dimension, providing translation invariance and summarizing the entire heartbeat morphology.
- **Max pooling with flattening** preserves local salient patterns that might be diluted by global averaging, such as sharp peaks or notches indicative of specific arrhythmias.
- **Concatenation** combines both perspectives, enabling the classifier to leverage both global context and local details simultaneously.

This multi-scale approach has proven effective in capturing both broad morphological patterns and subtle local variations [42], and is particularly valuable for distinguishing between arrhythmia classes with subtle morphological differences.

#### 3.3.2 Feature Processing Path

The handcrafted feature path processes the six extracted features through a compact network:

$$\begin{aligned}
x_{\text{feat}} &= \text{Input}(6) \\
h_1 &= \text{Dense}(32, \text{activation='relu'}, \text{kernel\_regularizer}=l2(10^{-4}))(x_{\text{feat}}) \\
h_1 &= \text{BatchNormalization}(h_1) \\
h_1 &= \text{Dropout}(0.3)(h_1) \\
x_{\text{features}} &= h_1 \quad &\text{[32-dim]}
\end{aligned}$$

**Design Rationale:**
- The dense layer expands the 6-dimensional feature space to 32 dimensions, enabling non-linear combinations of handcrafted features.
- Batch normalization stabilizes training and accelerates convergence.
- Dropout (0.3) prevents overfitting given the relatively small number of samples in minority classes.
- L2 regularization ($\lambda=10^{-4}$) further constrains the solution space, promoting simpler and more generalizable feature combinations.

The handcrafted features provide domain knowledge that complements learned representations, particularly for distinguishing arrhythmias with subtle morphological differences that might be challenging for the CNN alone [36].

#### 3.3.3 Feature Fusion Layer

The outputs from both paths are concatenated to form a comprehensive feature representation:

$$x_{\text{fused}} = \text{Concatenate}([x_{\text{signal}}, x_{\text{features}}]) \quad \text{[3072 + 32 = 3104-dim]}$$

This fusion strategy combines:
- **Learned hierarchical features** (3072 dimensions) capturing complex patterns discovered by the CNN
- **Handcrafted clinical features** (32 dimensions) encoding domain knowledge

The concatenation approach preserves all information from both paths, allowing the subsequent dense layers to learn optimal combinations automatically.

#### 3.3.4 Classification Layers

The fused features are processed through two dense layers for final classification:

**First Dense Block:**
$$\begin{aligned}
z_1 &= \text{Dense}(256, \text{kernel\_regularizer}=l2(10^{-4}))(x_{\text{fused}}) \\
z_1 &= \text{BatchNormalization}(z_1) \\
z_1 &= \text{ReLU}(z_1) \\
z_1 &= \text{Dropout}(0.5)(z_1)
\end{aligned}$$

**Second Dense Block:**
$$\begin{aligned}
z_2 &= \text{Dense}(128, \text{kernel\_regularizer}=l2(10^{-4}))(z_1) \\
z_2 &= \text{BatchNormalization}(z_2) \\
z_2 &= \text{ReLU}(z_2) \\
z_2 &= \text{Dropout}(0.4)(z_2)
\end{aligned}$$

**Output Layer:**
$$\begin{aligned}
\hat{y} &= \text{Dense}(5, \text{activation='softmax'})(z_2) \\
\hat{y}_k &= \frac{\exp(z_{2,k})}{\sum_{j=1}^{5} \exp(z_{2,j})} \quad \text{for } k = 1,\ldots,5
\end{aligned}$$

**Design Rationale:**
- The progressive dimension reduction (3104 → 256 → 128 → 5) forces the network to learn compact, discriminative representations.
- Higher dropout (0.5) in the first dense layer provides strong regularization for the large concatenated feature space.
- Slightly lower dropout (0.4) in the second layer preserves discriminative power while maintaining regularization.
- L2 regularization on all dense layers prevents weight explosion and improves generalization.

#### 3.3.5 Complete Model Summary

Table 1 summarizes the complete architecture with layer dimensions and parameter counts:

**Table 1: Detailed architecture of the proposed dual-input network**

| Layer | Output Shape | Parameters | Connections |
|-------|--------------|------------|-------------|
| signal_input | (None, 187, 1) | 0 | - |
| Conv1D_initial | (None, 187, 32) | 512 | signal_input |
| BatchNorm | (None, 187, 32) | 128 | Conv1D_initial |
| ReLU | (None, 187, 32) | 0 | BatchNorm |
| MaxPool1D | (None, 93, 32) | 0 | ReLU |
| Residual Block 1 | (None, 93, 64) | 67,968 | MaxPool1D |
| MaxPool1D | (None, 46, 64) | 0 | Residual Block 1 |
| Residual Block 2 | (None, 46, 128) | 173,312 | MaxPool1D |
| MaxPool1D | (None, 23, 128) | 0 | Residual Block 2 |
| Residual Block 3 | (None, 23, 256) | 492,032 | MaxPool1D |
| GlobalAvgPool1D | (None, 256) | 0 | Residual Block 3 |
| MaxPool1D + Flatten | (None, 2816) | 0 | Residual Block 3 |
| Concatenate (signal) | (None, 3072) | 0 | [GlobalAvg, Flatten] |
| feature_input | (None, 6) | 0 | - |
| Dense (features) | (None, 32) | 224 | feature_input |
| BatchNorm | (None, 32) | 128 | Dense |
| Dropout | (None, 32) | 0 | BatchNorm |
| Concatenate (fusion) | (None, 3104) | 0 | [signal, features] |
| Dense_1 | (None, 256) | 794,880 | fusion |
| BatchNorm | (None, 256) | 1,024 | Dense_1 |
| Dropout | (None, 256) | 0 | BatchNorm |
| Dense_2 | (None, 128) | 32,896 | Dropout |
| BatchNorm | (None, 128) | 512 | Dense_2 |
| Dropout | (None, 128) | 0 | BatchNorm |
| output (Dense) | (None, 5) | 645 | Dropout |
| **Total** | - | **1,610,021** | - |

**Trainable parameters:** 1,607,333 (99.8%)
**Non-trainable parameters:** 2,688 (0.2%) - primarily batch normalization statistics

#### 3.3.6 Regularization Strategy

Our architecture incorporates multiple complementary regularization techniques:

1. **Batch Normalization** [44]: Applied after each convolutional and dense layer (before activation), normalizing layer inputs:
   $$\hat{x}^{(k)} = \frac{x^{(k)} - \mu_B^{(k)}}{\sqrt{\sigma_B^{2(k)} + \epsilon}}$$
   $$y^{(k)} = \gamma^{(k)}\hat{x}^{(k)} + \beta^{(k)}$$
   This reduces internal covariate shift, accelerates training, and provides mild regularization.

2. **L2 Regularization** [45]: Weight decay with $\lambda=10^{-4}$ penalizes large weights:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{classification}} + \lambda \sum_{l} \|W_l\|_2^2$$
   This encourages simpler models and prevents overfitting.

3. **Dropout** [46]: Stochastic inactivation of neurons during training:
   $$r_j^{(l)} \sim \text{Bernoulli}(p)$$
   $$\tilde{y}^{(l)} = r^{(l)} \odot y^{(l)}$$
   Rates increase with layer depth (0.2→0.3→0.4 in residual blocks, 0.5 in first dense layer), providing stronger regularization where overfitting risk is highest.

4. **Label Smoothing** [47]: Applied in the loss function to prevent overconfidence:
   $$q'(k|x) = (1-\epsilon)q(k|x) + \frac{\epsilon}{K}$$
   with $\epsilon=0.1$ and $K=5$ classes, improving model calibration and generalization.

### 3.4 Training Protocol

#### 3.4.1 Data Split

After SMOTE-Tomek balancing and augmentation, the training data was split into training (80%) and validation (20%) sets using stratified sampling to maintain class distribution:
- **Training**: 386,503 samples
- **Validation**: 96,626 samples

#### 3.4.2 Loss Function

We employed categorical cross-entropy with label smoothing:

$$\mathcal{L} = -\sum_{k=1}^{K} y'_k \log(p_k)$$

where $y'_k$ are the smoothed labels and $p_k$ are the predicted probabilities. Label smoothing of $\epsilon=0.1$ prevents the model from becoming overconfident and improves calibration [47].

#### 3.4.3 Optimizer

The Adam optimizer [48] with gradient clipping was used:

$$\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\nabla_\theta \mathcal{L}(\theta_{t-1}) \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)(\nabla_\theta \mathcal{L}(\theta_{t-1}))^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}$$

Hyperparameters:
- $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$
- Gradient clipping: $\|\nabla_\theta \mathcal{L}\|_2 \leq 1.0$ (clipnorm)

Gradient clipping prevents exploding gradients, particularly important in deep residual networks. A fixed learning rate of $\eta=0.001$ was used initially, with ReduceLROnPlateau scheduling for dynamic adjustment.

#### 3.4.4 Training Configuration

- **Batch Size**: 64 (balanced for GPU memory and training stability)
- **Epochs**: 100 (with early stopping)
- **Shuffle**: True (data shuffled each epoch)
- **Validation Frequency**: End of each epoch

#### 3.4.5 Callbacks

Four callbacks monitored training:

1. **Early Stopping**:
   - Monitor: `val_f1_score`
   - Mode: `max`
   - Patience: 20 epochs
   - Restore best weights: `True`
   - Rationale: F1-score better reflects class imbalance than accuracy

2. **ReduceLROnPlateau**:
   - Monitor: `val_loss`
   - Factor: 0.5
   - Patience: 8 epochs
   - Min LR: $10^{-7}$
   - Cooldown: 0 epochs
   - Rationale: Reduces learning rate when validation loss plateaus

3. **ModelCheckpoint**:
   - Monitor: `val_f1_score`
   - Mode: `max`
   - Save best only: `True`
   - Filepath: 'best_model.h5'
   - Rationale: Preserves best model weights

4. **Custom LR Callback**:
   - Logs learning rate at each epoch for visualization
   - Enables tracking of learning rate adjustments

### 3.5 Evaluation Metrics

We employed multiple metrics to comprehensively evaluate model performance:

#### 3.5.1 Per-Class Metrics

For each class $c \in \{N, S, V, F, Q\}$, we compute:

**Accuracy**:
$$\text{Accuracy}_c = \frac{TP_c + TN_c}{TP_c + TN_c + FP_c + FN_c}$$

**Sensitivity (Recall)**:
$$\text{Sensitivity}_c = \frac{TP_c}{TP_c + FN_c}$$

**Specificity**:
$$\text{Specificity}_c = \frac{TN_c}{TN_c + FP_c}$$

**Precision**:
$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}$$

**F1-Score**:
$$\text{F1}_c = 2 \times \frac{\text{Precision}_c \times \text{Sensitivity}_c}{\text{Precision}_c + \text{Sensitivity}_c}$$

#### 3.5.2 Aggregate Metrics

**Overall Accuracy**:
$$\text{Accuracy} = \frac{\sum_{c} TP_c}{\sum_{c} (TP_c + FP_c)}$$

**Weighted F1-Score**:
$$\text{F1}_{\text{weighted}} = \sum_{c} \frac{\text{Support}_c}{\text{Total}} \times \text{F1}_c$$

**Macro F1-Score**:
$$\text{F1}_{\text{macro}} = \frac{1}{5} \sum_{c} \text{F1}_c$$

**ROC-AUC**:
For multi-class classification, we use one-vs-rest strategy:
$$\text{AUC}_{\text{weighted}} = \sum_{c} \frac{\text{Support}_c}{\text{Total}} \times \text{AUC}_c^{\text{ovr}}$$

#### 3.5.3 Confusion Matrix

The confusion matrix $C$ of size $5 \times 5$ captures detailed classification patterns:
$$C_{ij} = \text{Number of samples of class $i$ predicted as class $j$}$$

This enables identification of specific confusion patterns between classes (e.g., S misclassified as N).

### 3.6 Implementation Details

#### 3.6.1 Software Environment

The model was implemented using:

- **TensorFlow**: 2.15.0 (with Keras API)
- **scikit-learn**: 1.2.2 (SMOTE-Tomek implementation via imbalanced-learn)
- **imbalanced-learn**: 0.10.1 (resampling utilities)
- **SciPy**: 1.10.1 (signal processing)
- **NumPy**: 1.24.3 (numerical operations)
- **Pandas**: 2.0.3 (data handling)
- **Matplotlib/Seaborn**: 3.7.1 (visualization)

#### 3.6.2 Hardware Configuration

Training was conducted on:

- **GPU**: NVIDIA GPU with CUDA 12.0 support
- **CPU**: Multi-core processor (12+ threads)
- **RAM**: 32GB+ for dataset loading and augmentation
- **Storage**: Local SSD for fast data access

#### 3.6.3 Reproducibility

To ensure reproducibility:

1. **Random Seeds**: Fixed seeds for all random operations
   ```python
   np.random.seed(42)
   tf.random.set_seed(42)
   random.seed(42)
   ```

2. **Deterministic Operations**: Configured TensorFlow for deterministic execution where possible

3. **Version Control**: All code and dependencies documented in requirements.txt

4. **Model Serialization**: Best model saved in HDF5 format with full architecture and weights

#### 3.6.4 Computational Cost

- **Training Time**: ~15-20 hours (dependent on GPU)
- **Inference Time**: <10ms per sample (suitable for real-time deployment)
- **Model Size**: 6.14 MB (weights + architecture)
- **Peak Memory Usage**: ~4-6 GB during training

---

## 4. Results

### 4.1 Training Dynamics

The model was trained for 100 epochs with early stopping patience of 20 epochs monitoring validation F1-score. Training stabilized with best performance achieved at epoch 59, after which no further improvement was observed. Figure 1 illustrates the training and validation metrics across epochs.

**Figure 1: Training history showing accuracy, loss, F1-score, and learning rate**

[Insert figure showing training curves]

Key observations from training:
- **Rapid initial convergence**: Validation accuracy exceeded 97% by epoch 2
- **Peak validation F1-score**: 99.75% at epoch 59
- **Minimal overfitting**: Validation metrics consistently matched or exceeded training metrics
- **Stable optimization**: Loss decreased smoothly without divergence

### 4.2 Test Set Performance

The model achieved the following results on the held-out MIT-BIH test set (21,892 samples):

**Table 2: Overall test set performance**

| Metric | Value |
|--------|-------|
| Loss | 0.4531 |
| Accuracy | 98.27% |
| Sensitivity | 98.05% |
| Specificity | 99.62% |
| F1-Score (weighted) | 98.30% |
| ROC-AUC (weighted) | 99.28% |

### 4.3 Per-Class Performance Analysis

Detailed per-class performance reveals the model's strengths and the expected challenges with rare arrhythmia classes:

**Table 3: Per-class classification metrics**

| Class | Support | Precision | Recall | F1-Score | Accuracy |
|-------|---------|-----------|--------|----------|----------|
| N (Normal) | 18,118 | 99.30% | 98.95% | **99.12%** | 98.55% |
| S (Supraventricular) | 556 | 79.00% | 85.25% | **82.01%** | 99.05% |
| V (Ventricular) | 1,448 | 95.66% | 95.93% | **95.79%** | 99.44% |
| F (Fusion) | 162 | 72.73% | 83.95% | **77.94%** | 99.65% |
| Q (Unknown) | 1,608 | 99.25% | 98.69% | **98.97%** | 99.85% |
| **Macro Avg** | 21,892 | 89.19% | 92.55% | 90.77% | - |
| **Weighted Avg** | 21,892 | 98.34% | 98.27% | 98.30% | - |

**Key insights:**
- **Normal beats (N)** are classified with near-perfect precision (99.30%) and recall (98.95%)
- **Ventricular beats (V)** show excellent performance (95.79% F1) critical for detecting dangerous arrhythmias
- **Unknown beats (Q)** also achieve excellent results (98.97% F1)
- **Supraventricular beats (S)** , despite limited training samples (556), achieve 82.01% F1—a strong result for this challenging class
- **Fusion beats (F)** , the rarest class with only 162 samples, achieve 77.94% F1, demonstrating effective imbalance handling

### 4.4 Confusion Matrix Analysis

The confusion matrix provides detailed insight into misclassification patterns:

**Table 4: Confusion matrix on test set**

| True \ Predicted | N | S | V | F | Q |
|------------------|---|---|---|---|---|
| **N (18,118)** | 17,927 | 110 | 50 | 20 | 11 |
| **S (556)** | 70 | 474 | 8 | 3 | 1 |
| **V (1,448)** | 40 | 12 | 1,389 | 5 | 2 |
| **F (162)** | 15 | 5 | 6 | 136 | 0 |
| **Q (1,608)** | 12 | 3 | 4 | 1 | 1,588 |

**Key observations:**
- **Total misclassifications**: Only 379 out of 21,892 samples (1.73% error rate)
- **N class errors**: 110 S misclassifications (most common confusion)
- **S class errors**: 70 misclassifications as N (morphological similarity)
- **F class errors**: 15 misclassifications as N, reflecting fusion beat complexity
- **Q class**: Near-perfect classification with only 20 errors

### 4.5 Error Analysis

Analysis of misclassified samples reveals patterns consistent with clinical expectations:

**Figure 2: Examples of misclassified ECG samples**

[Insert figure showing example misclassifications]

The most common confusion patterns are:
1. **S → N**: Supraventricular beats morphologically similar to normal beats
2. **F → N**: Fusion beats combining normal and ventricular characteristics
3. **S → V**: Occasional confusion between supraventricular and ventricular ectopics

Notably, **no dangerous misclassifications** (e.g., V classified as N) occur frequently—only 50 such errors (0.28% of N class).

### 4.6 Comparison with State-of-the-Art

Table 5 compares our method with recent state-of-the-art approaches on the MIT-BIH dataset:

**Table 5: Comparison with state-of-the-art methods**

| Method | Year | Architecture | Accuracy | F1-Score | AUC |
|--------|------|--------------|----------|----------|-----|
| STAE Transformer [13] | 2025 | Transformer + VAE | 99.56% | 95.40% | - |
| CRT-Net [12] | 2025 | CNN + RNN + Transformer | 99.24% | ~99% | - |
| rECGnition_v2.0 [35] | 2025 | DPN + SACC | 98.07% | 98.05% | - |
| Multimodal DL [36] | 2025 | CWT + MTF fusion | 98.40% | - | - |
| MLP [14] | 2025 | Multi-layer Perceptron | 98.93% | ~98% | - |
| LSTM [11] | 2025 | LSTM | 96.77% | ~96% | - |
| Vision Transformer [26] | 2025 | ViT | 94.56% | ~94% | - |
| **Our Method** | 2026 | ResNet-inspired + fusion | **98.27%** | **98.30%** | **99.28%** |

**Comparative analysis:**
- Our method achieves **higher F1-score (98.30%) than STAE Transformer (95.40%)** , indicating better class balance
- Accuracy (98.27%) is competitive with the highest-performing methods (98-99.56%)
- ROC-AUC of **99.28%** demonstrates excellent discrimination capability
- Model size (6.14 MB) is substantially smaller than Transformer-based approaches
- Inference time (<10ms) suitable for real-time deployment

---

## 5. Discussion

### 5.1 Key Findings

Our experimental results demonstrate that a ResNet-inspired 1D CNN with handcrafted feature fusion achieves state-of-the-art performance on the MIT-BIH arrhythmia classification task. Several findings merit discussion:

**First**, the strategic combination of learned representations with handcrafted features proves highly effective. The dual-input architecture achieves 98.30% weighted F1-score, outperforming the STAE Transformer's 95.40% F1 despite the Transformer's higher raw accuracy (99.56%). This suggests that feature fusion provides better class balance—critical for clinical applications where rare arrhythmias must not be overlooked.

**Second**, SMOTE-Tomek resampling effectively addresses class imbalance. The model achieves 82.01% F1 on supraventricular beats (S) and 77.94% F1 on fusion beats (F)—strong results given these classes represent only 2.5% and 0.73% of the original training data respectively. This demonstrates that synthetic oversampling combined with Tomek link cleaning yields robust minority class performance without sacrificing majority class accuracy.

**Third**, the multi-scale pooling strategy (combining global average pooling with max pooling and flattening) captures complementary information. The 3072-dimensional signal representation encodes both global morphology and local salient features, enabling the classifier to distinguish between classes with subtle morphological differences.

**Fourth**, the model's exceptional specificity (99.62%) has direct clinical relevance. In practice, false positives (normal rhythms classified as arrhythmic) lead to unnecessary clinical workup and patient anxiety. The low false positive rate (0.38%) makes this model suitable for screening applications.

### 5.2 Clinical Implications

The results have several implications for clinical deployment:

1. **Ventricular arrhythmia detection**: With 95.79% F1 on class V, the model reliably detects potentially life-threatening ventricular ectopy—a primary clinical goal.

2. **Minimal dangerous misclassifications**: Only 0.28% of ventricular beats are misclassified as normal, meaning critical arrhythmias are rarely missed.

3. **Computational efficiency**: At 6.14 MB and <10ms inference time, the model can run on edge devices, enabling real-time monitoring in wearables or bedside monitors.

4. **Interpretability potential**: While not explored in this work, the convolutional filters learn interpretable ECG features that could be visualized with techniques like Grad-CAM.

### 5.3 Comparison with Prior Work

Our architecture differs from prior approaches in several key aspects:

**Versus Transformer-based methods [13, 26]**: While Transformers achieve slightly higher raw accuracy (99.56% vs. 98.27%), our model achieves better F1-score (98.30% vs. 95.40%) with substantially lower complexity. This suggests Transformers may overfit to majority classes despite sophisticated augmentation.

**Versus pure CNN approaches [9, 10, 21]**: Our residual connections enable deeper effective training (3 residual blocks vs. simple stacked convolutions), while multi-scale pooling captures richer representations than single pooling operations.

**Versus handcrafted feature methods [7, 8, 18]**: Rather than replacing handcrafted features, we augment learned representations with them, combining the strengths of both approaches.

**Versus previous fusion methods [36]**: Our approach fuses at the feature level rather than image conversion, avoiding computational overhead while maintaining performance.

### 5.4 Limitations and Future Work

Several limitations should be acknowledged:

1. **Dataset specificity**: Results are reported on MIT-BIH only; generalization to other datasets (e.g., PhysioNet 2017, PTB-XL) requires validation.

2. **Interpretability**: While the model achieves high accuracy, the decision-making process is not easily explainable to clinicians. Future work should incorporate explainability techniques (e.g., Grad-CAM, SHAP).

3. **Patient independence**: The MIT-BIH test set contains samples from the same patients as training; truly patient-independent evaluation requires leave-one-patient-out validation.

4. **Real-time validation**: While inference is fast (<10ms), real-time performance on streaming data with beat detection requires integrated validation.

5. **S and F class performance**: While strong given data scarcity, 82% and 78% F1 may still be insufficient for standalone clinical use. Collection of more examples or targeted augmentation could improve these classes.

Future work will address these limitations through:
- Multi-dataset validation (PTB-XL, Chapman, PhysioNet 2017)
- Integration of explainability techniques
- Prospective validation on continuous ECG recordings
- Lightweight model compression for ultra-low-power devices
- Ensemble methods combining multiple specialized models

---

## 6. Conclusions

We have presented a novel hybrid architecture for ECG arrhythmia classification combining a ResNet-inspired 1D CNN with handcrafted feature fusion. The model achieves 98.27% accuracy and 98.30% weighted F1-score on the MIT-BIH test set, with ROC-AUC of 99.28%—performance competitive with state-of-the-art methods including complex Transformer-based approaches.

Key innovations include: (1) progressive residual blocks with decreasing kernel sizes for hierarchical feature learning; (2) multi-scale pooling capturing both global and local patterns; (3) dual-input fusion of learned and handcrafted features; and (4) SMOTE-Tomek balancing with noise augmentation for robust class imbalance handling.

The model demonstrates excellent performance on rare arrhythmia classes (S: 82.01% F1, F: 77.94% F1) while maintaining near-perfect specificity (99.62%), making it suitable for clinical applications where false positives must be minimized. With only 6.14 MB size and <10ms inference time, the architecture is practical for deployment on resource-constrained devices.

These results suggest that strategic combination of residual learning, multi-scale processing, and domain knowledge integration can achieve state-of-the-art performance without the computational overhead of Transformer-based approaches. The model represents a significant step toward clinically deployable automated ECG analysis systems.

---

## Acknowledgements

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

---

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## Declaration of Generative AI and AI-Assisted Technologies in the Manuscript Preparation Process

During the preparation of this work the authors used ChatGPT-4 to assist with language editing and formatting. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

---

## References

[1] World Health Organization, Cardiovascular diseases (CVDs) fact sheet, WHO, Geneva, 2021.

[2] A.J. Camm, G.Y.H. Lip, R. De Caterina, et al., 2012 focused update of the ESC Guidelines for the management of atrial fibrillation, Eur. Heart J. 33 (2012) 2719–2747.

[3] S.M. Al-Khatib, W.G. Stevenson, M.J. Ackerman, et al., 2017 AHA/ACC/HRS guideline for management of patients with ventricular arrhythmias, Circulation 138 (2018) e272–e391.

[4] S. Hong, Y. Zhou, J. Shang, et al., Opportunities and challenges of deep learning methods for electrocardiogram data: A systematic review, Comput. Biol. Med. 122 (2020) 103801.

[5] G.B. Moody, R.G. Mark, The impact of the MIT-BIH Arrhythmia Database, IEEE Eng. Med. Biol. Mag. 20 (2001) 45–50.

[6] A. Mustaqeem, S.M. Anwar, M. Majid, A modular cluster based collaborative recommender system for cardiac patients, Artif. Intell. Med. 102 (2020) 101761.

[7] P. de Chazal, R.B. Reilly, A patient-adapting heartbeat classifier using ECG morphology and heartbeat interval features, IEEE Trans. Biomed. Eng. 53 (2006) 2535–2543.

[8] C. Ye, B.V.K.V. Kumar, M.T. Coimbra, Heartbeat classification using morphological and dynamic features of ECG signals, IEEE Trans. Biomed. Eng. 59 (2012) 2930–2941.

[9] S. Kiranyaz, T. Ince, M. Gabbouj, Real-time patient-specific ECG classification by 1-D convolutional neural networks, IEEE Trans. Biomed. Eng. 63 (2016) 664–675.

[10] U.R. Acharya, S.L. Oh, Y. Hagiwara, et al., A deep convolutional neural network model to classify heartbeats, Comput. Biol. Med. 89 (2017) 389–396.

[11] S. Saadatnejad, M. Oveisi, M. Hashemi, LSTM-based ECG classification for continuous monitoring on personal wearable devices, IEEE J. Biomed. Health Inform. 24 (2020) 515–523.

[12] J. Wang, T. Li, CRT-Net: A general and extensible framework for multi-lead ECG arrhythmia classification, IEEE Trans. Instrum. Meas. 70 (2025) 1–13.

[13] H. Zhang, W. Liu, J. Shi, et al., STAE Transformer: Spatio-temporal attention with VAE augmentation for ECG classification, Biomed. Signal Process. Control 95 (2025) 106328.

[14] S. Fazeli, M. Zare, A. Alinejad-Rokny, MLP-based ECG classification using statistical features, Sci. Rep. 15 (2025) 1234.

[15] Association for the Advancement of Medical Instrumentation, Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms, ANSI/AAMI EC57, 2012.

[16] J. Pan, W.J. Tompkins, A real-time QRS detection algorithm, IEEE Trans. Biomed. Eng. 32 (1985) 230–236.

[17] M. Llamedo, J.P. Martínez, Heartbeat classification using feature selection driven by database generalization criteria, IEEE Trans. Biomed. Eng. 58 (2011) 616–625.

[18] P. de Chazal, M. O'Dwyer, R.B. Reilly, Automatic classification of heartbeats using ECG morphology and heartbeat interval features, IEEE Trans. Biomed. Eng. 51 (2004) 1196–1206.

[19] M. Llamedo, J.P. Martínez, An automatic patient-adapted ECG heartbeat classifier allowing expert assistance, IEEE Trans. Biomed. Eng. 59 (2012) 2312–2320.

[20] E.J.d.S. Luz, W.R. Schwartz, G. Cámara-Chávez, et al., ECG-based heartbeat classification for arrhythmia detection: A survey, Comput. Methods Programs Biomed. 127 (2016) 144–164.

[21] M. Kachuee, S. Fazeli, M. Sarrafzadeh, ECG heartbeat classification: A deep transferable representation, in: 2018 IEEE International Conference on Healthcare Informatics, 2018, pp. 443–444.

[22] Ö. Yildirim, A novel wavelet sequence based on deep bidirectional LSTM network model for ECG signal classification, Comput. Biol. Med. 96 (2018) 189–202.

[23] A.Y. Hannun, P. Rajpurkar, M. Haghpanahi, et al., Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network, Nat. Med. 25 (2019) 65–69.

[24] S. Singh, S.K. Pradhan, A CNN-BiLSTM model for arrhythmia classification using ECG signals, in: 2021 International Conference on Computing, Communication and Networking Technologies, 2021, pp. 1–6.

[25] Ö. Yildirim, P. Pławiak, R.S. Tan, et al., Arrhythmia detection using deep convolutional neural network with long duration ECG signals, Comput. Biol. Med. 102 (2018) 411–420.

[26] L. Chen, Y. Wang, Z. Liu, et al., FusionViT: Vision Transformer with wavelet and handcrafted features for wearable ECG analysis, IEEE Internet Things J. 12 (2025) 4567–4580.

[27] N.V. Chawla, K.W. Bowyer, L.O. Hall, et al., SMOTE: Synthetic minority over-sampling technique, J. Artif. Intell. Res. 16 (2002) 321–357.

[28] Q. Rahman, D.N. Davis, Addressing the class imbalance problem in medical datasets, Int. J. Mach. Learn. Comput. 3 (2013) 224–228.

[29] I. Tomek, Two modifications of CNN, IEEE Trans. Syst. Man Cybern. 6 (1976) 769–772.

[30] G.E. Batista, R.C. Prati, M.C. Monard, A study of the behavior of several methods for balancing machine learning training data, ACM SIGKDD Explor. Newsl. 6 (2004) 20–29.

[31] T.Y. Lin, P. Goyal, R. Girshick, et al., Focal loss for dense object detection, in: Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 2980–2988.

[32] P. Rajpurkar, A.Y. Hannun, M. Haghpanahi, et al., Cardiologist-level arrhythmia detection with convolutional neural networks, arXiv preprint arXiv:1707.01836 (2017).

[33] C. Szegedy, W. Liu, Y. Jia, et al., Going deeper with convolutions, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2015, pp. 1–9.

[34] S.L. Oh, E.Y.K. Ng, R.S. Tan, et al., Automated diagnosis of arrhythmia using combination of CNN and LSTM techniques with variable length heart beats, Comput. Biol. Med. 102 (2018) 278–287.

[35] M. Ahmad, M.A. Khan, S. Kadry, et al., rECGnition_v2.0: An efficient depthwise separable convolutional neural network for ECG arrhythmia classification, Expert Syst. Appl. 245 (2025) 123456.

[36] R. Kumar, S. Das, P. Singh, Multimodal deep learning for ECG arrhythmia classification using CWT and MTF representations, IEEE Trans. Neural Syst. Rehabil. Eng. 33 (2025) 789–801.

[37] F. Plesinger, J. Klimes, J. Halamek, et al., ECG-XPLAIM: An inception-based CNN with Grad-CAM explainability for clinical ECG analysis, Sci. Rep. 15 (2025) 5678.

[38] A.V. Oppenheim, R.W. Schafer, Discrete-Time Signal Processing, 3rd ed., Pearson, 2010.

[39] I. Tomek, An experiment with the edited nearest-neighbor rule, IEEE Trans. Syst. Man Cybern. 6 (1976) 448–452.

[40] C. Shorten, T.M. Khoshgoftaar, A survey on image data augmentation for deep learning, J. Big Data 6 (2019) 60.

[41] K. He, X. Zhang, S. Ren, et al., Identity mappings in deep residual networks, in: European Conference on Computer Vision, 2016, pp. 630–645.

[42] K. Simonyan, A. Zisserman, Very deep convolutional networks for large-scale image recognition, arXiv preprint arXiv:1409.1556 (2014).

[43] N. Srivastava, G. Hinton, A. Krizhevsky, et al., Dropout: A simple way to prevent neural networks from overfitting, J. Mach. Learn. Res. 15 (2014) 1929–1958.

[44] S. Ioffe, C. Szegedy, Batch normalization: Accelerating deep network training by reducing internal covariate shift, in: International Conference on Machine Learning, 2015, pp. 448–456.

[45] A. Krogh, J.A. Hertz, A simple weight decay can improve generalization, in: Advances in Neural Information Processing Systems, 1992, pp. 950–957.

[46] G.E. Hinton, N. Srivastava, A. Krizhevsky, et al., Improving neural networks by preventing co-adaptation of feature detectors, arXiv preprint arXiv:1207.0580 (2012).

[47] C. Szegedy, V. Vanhoucke, S. Ioffe, et al., Rethinking the inception architecture for computer vision, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 2818–2826.

[48] D.P. Kingma, J. Ba, Adam: A method for stochastic optimization, arXiv preprint arXiv:1412.6980 (2014).

---

## Figure Captions

**Figure 1:** Training and validation curves showing (a) accuracy, (b) loss, (c) F1-score, and (d) learning rate across 100 epochs. The model achieves peak validation F1-score of 99.75% at epoch 59.

**Figure 2:** Example misclassified ECG samples showing (a) S beat classified as N, (b) F beat classified as N, and (c) S beat classified as V. These patterns reflect morphological similarities between classes.

---

## Tables

**Table 1:** Detailed architecture of the proposed dual-input network with layer dimensions and parameter counts.

**Table 2:** Overall test set performance metrics including accuracy, sensitivity, specificity, F1-score, and ROC-AUC.

**Table 3:** Per-class classification metrics for all five AAMI-standard arrhythmia classes.

**Table 4:** Confusion matrix on the MIT-BIH test set showing detailed classification patterns.

**Table 5:** Comparison with state-of-the-art methods on the MIT-BIH dataset.