# «Quantitative Evaluation of XAI Methods for Image Segmentation in Autonomous Driving»
## Bachelor Thesis - Liridona Cerkini - Rotkreuz, May 30th, 2025

## Project Overview
### Description 
This project evaluates various Explainable AI (XAI) methods in the context of semantic segmentation for autonomous driving. The evaluation is divided into three phases:

- Phase 1: Model application using uncompressed high-resolution images from the Mapillary Vistas dataset, segmented using OneFormer. Segmentation accuracy is evaluated using the Intersection-over-Union (IoU) metric.
- Phase 2: Application of XAI methods (LIME, Grad-CAM, Seg-Grad-CAM, Guided Grad-CAM, Saliency, XRAI, and L-CRP) to analyze interpretability.
- Phase 3: Systematic evaluation of XAI methods using quantitative metrics (e.g. IROF, Max-Sensitivity, Focus, Pointing Game, Effective Complexity) targeting robustness, faithfulness, localization accuracy, and explanation complexity.

### Project Goals
- Given the initial circumstances and objectives, the following primary research questions and two sub-question were developed:
-	Which methods for image segmentation perform well on universal models, particularly in the context of autonomous driving?
-	What is the performance of the universal model in semantic segmentation tasks?
-	How do methods perform in terms of established evaluation metrics?
 
### Data
- Dataset: Mapillary Vistas (15,000 images)
- Models Used: OneFormer for segmentation; DeeplabV3 in later evaluation stages
- Tasks: Semantic segmentation on selected labels with similar frequency distribution

### Approach
#### Model & Segmentation
- Model: OneFormer
- Dataset preprocessing and segmentation using HuggingFace transformers and torchvision

#### XAI Methods Evaluated
- LIME (Captum)
- Saliency (Vanilla Gradient - Captum)
- Grad-CAM (LayerGradCAM - Captum)
- Seg-Grad-CAM (GitHub)
- XRAI (Google Saliency)
- L-CRP (GitHub)
- Integrated Gradients (Captum)

#### Evaluation Metrics (used in Phase 3)
- Max-Sensitivity (Yeh et al., 2019): Measures the stability of explanations under slight input perturbations (robustness).
- IROF (Rieger & Hansen, 2020): Evaluates the faithfulness of the explanation by assessing the impact of feature removal on classification accuracy.
- Pointing Game (Zhang et al., 2016): Assesses localization accuracy by verifying if the highest attribution score corresponds to the target object.
- Focus (Arias-Duart et al., 2022): A localization metric that identifies misfocus errors and reveals model biases using mosaic data.
- Effective Complexity (Nguyen & Martínez, 2020): Measures how many features are necessary to explain the prediction without deviating significantly from model accuracy.

### Data Used
- Cordts u. a., „The Cityscapes Dataset for Semantic Urban Scene Understanding“. (Quelle: http://arxiv.org/abs/1604.01685)
  
### Additional Sources
- DeeplabV3: https://arxiv.org/pdf/1706.05587
- OneFormer: http://arxiv.org/abs/2211.06220
- HuggingFace: https://huggingface.co/
- DeeplabV3 Benchmark (via Paperswithcode): https://paperswithcode.com/method/deeplabv3
- OneFormer Benchmark (via Paperswithcode): https://paperswithcode.com/paper/oneformer-one-transformer-to-rule-universal

## Project Structure 
- Model_Liridona_C.ipynb: Model application and segmentation (IoU evaluation)
- Methods&Metrics_Liridona_C.ipynb: XAI method application and evaluation
- README.md: Project documentation
  
## Installation & Environment
### Recommended Environment
- PyTorch 2.3.1 with GPU support
- CUDA 12.1
### Package Installation
*pip install transformers*
*pip install opencv-python*
*pip install --upgrade grad-cam*
*pip install captum*
*pip install saliency[tf1]*
*pip install quantus*
*pip install cachetools*
### Libraries Used
- PyTorch, TorchVision – model and image handling
- Captum, Grad-CAM, Google Saliency, Quantus – XAI methods and metrics
- NumPy, Pandas, Matplotlib, TQDM – data handling and visualization
- OpenCV, PIL, scikit-image – image processing
- Transformers – OneFormer model integration

## Instructions for Running the Code
1. Start the environment: Open “GPU Hub” and launch a Jupyter notebook server with “PyTorch 2.3.1 and GPU support.
2. Clone this repository:
git clone https://github.com/woahitsdona/Quantitative-Evaluation-of-XAI-Methods-for-Image-Segmentation-in-Autonomous-Driving.git
cd Quantitative-Evaluation-of-XAI-Methods-for-Image-Segmentation-in-Autonomous-Driving

3. Run the notebook: Open the notebook Individualprojekt_AI_HS24_Liridona_C.ipynb in JupyterLab and execute the cells sequentially to prepare the data, train the models, and analyze the results.
https://github.com/woahitsdona/AI_Individualprojekt.git
## Citation & Sources
- Neuhold et al., 2017: Mapillary Vistas Dataset
- OneFormer: arXiv
- Benchmarks: Paperswithcode
- XAI Repos: Captum, Seg-Grad-CAM, L-CRP, Google Saliency
- Metrics: Quantus GitHub

## Notes
- This repository includes two notebooks documenting the full pipeline.
- For reproducibility, all random seeds are fixed, and warnings are suppressed.
- The results are shared in the submitted thesis report via ILIAS.
