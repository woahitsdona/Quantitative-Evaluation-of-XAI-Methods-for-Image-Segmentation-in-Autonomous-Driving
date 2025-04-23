# «Quantitative Evaluation of XAI Methods for Image Segmentation in Autonomous Driving»
## Bachelor Thesis - Liridona Cerkini - Rotkreuz, May 30th, 2025

## Project Overview
### Description 
This project evaluates 6 Explainable AI (XAI) methods in the context of semantic segmentation for autonomous driving. The evaluation is divided into three phases:

- Phase 1: Model application using uncompressed high-resolution images from the Mapillary Vistas dataset, segmented using OneFormer. Segmentation accuracy is evaluated using the Intersection-over-Union (IoU) metric.
- Phase 2: Application of XAI methods (LIME, Grad-CAM, Seg-Grad-CAM, Saliency (Vanilla Gradient), XRAI, Integrated Gradients (IG)) to analyze interpretability.
- Phase 3: Systematic evaluation of XAI methods using quantitative metrics (e.g. IROF, Max-Sensitivity, Focus, Pointing Game, Effective Complexity) targeting robustness, faithfulness, localization accuracy, and explanation complexity.

### Project Goals
Formulated in the preliminary study, the following primary research question and two sub-questions guide the analysis presented in the notebooks:
-	Which methods for image segmentation perform well on universal models, particularly in the context of autonomous driving?
  - What is the performance of the universal model in semantic segmentation tasks?
  - How do methods perform in terms of established evaluation metrics?
 
### Data
- Dataset: Mapillary Vistas (subset of 805 images with uniform dimensions (3264×2448 pixels))
  (Neuhold, G., Ollmann, T., Bulo, S. R., & Kontschieder, P. (2017). The Mapillary Vistas Dataset for Semantic Understanding of Street Scenes. 2017 IEEE International Conference on Computer Vision (ICCV), 5000–5009. https://doi.org/10.1109/ICCV.2017.534)
- Models Used: OneFormer for segmentation; DeeplabV3 in later evaluation stages
- Tasks: Semantic segmentation on selected labels

### Approach
#### Model & Segmentation
- Model: OneFormer and DeeplabV3
- Dataset preprocessing and segmentation using HuggingFace transformers and torchvision

#### XAI Methods Evaluated
- **LIME**, **Saliency**, and **Grad-CAM** via Captum: https://github.com/pytorch/captum
- **Seg-Grad-CAM**: https://github.com/kiraving/SegGradCAM
- **XRAI** via Google’s Saliency Library: https://github.com/PAIR-code/saliency
- **Integrated Gradients**: https://github.com/ankurtaly/Integrated-Gradients

#### Evaluation Metrics (used in Phase 3)
- **Max-Sensitivity** (Yeh et al., 2019): Measures the stability of explanations under slight input perturbations (robustness).
- **IROF** (Rieger & Hansen, 2020): Evaluates the faithfulness of the explanation by assessing the impact of feature removal on classification accuracy.
- **Pointing Game** (Zhang et al., 2016): Assesses localization accuracy by verifying if the highest attribution score corresponds to the target object.
- **Focus** (Arias-Duart et al., 2022): A localization metric that identifies misfocus errors and reveals model biases using mosaic data.
- **Effective Complexity** (Nguyen & Martínez, 2020): Measures how many features are necessary to explain the prediction without deviating significantly from model accuracy.
  
### Additional Sources
- OneFormer: http://arxiv.org/abs/2211.06220
- DeeplabV3: https://arxiv.org/pdf/1706.05587
- HuggingFace: https://huggingface.co/shi-labs/oneformer_coco_swin_large
- DeeplabV3 Benchmark (via Paperswithcode): https://paperswithcode.com/method/deeplabv3
- OneFormer Benchmark (via Paperswithcode): https://paperswithcode.com/paper/oneformer-one-transformer-to-rule-universal
- Quantus XAI Toolkit (Hedström et al., 2023): https://github.com/understandable-machine-intelligence-lab/Quantus

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
2. Clone this repository: git clone https://github.com/woahitsdona/Quantitative-Evaluation-of-XAI-Methods-for-Image-Segmentation-in-Autonomous-Driving.git
cd Quantitative-Evaluation-of-XAI-Methods-for-Image-Segmentation-in-Autonomous-Driving
3. Run the notebook: Open Model_Liridona_C.ipynb or Methods&Metrics_Liridona_C.ipynb in JupyterLab and execute the cells sequentially to preprocess the data, apply the models and XAI methods, and perform the corresponding analyses.

## Notes
- This repository includes two notebooks documenting the full pipeline.
- For reproducibility, all random seeds are fixed, and warnings are suppressed.
- The results are shared in the submitted thesis report via ILIAS.
