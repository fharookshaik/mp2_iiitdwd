# Multimodal Content Annotation for Identification of Comic Mischief

### Overview

This repository contains the project report and associated materials for the Mini Project II titled "Multimodal Content Annotation for Identification of Comic Mischief", submitted as part of the B.Tech degree requirements at the Indian Institute of Information Technology Dharwad (IIIT Dharwad). The project was conducted by a team of fourth-year Computer Science and Engineering students under the guidance of **Dr. Sunil Saumya**, Assistant Professor, Dept. of CSE, IIIT Dharwad, from August 2021 to November 2021.


|Name|Regd. No|
|-|-|
|Sumith Sai Budde| 18BCS101|
|Shaik Fharook| 18BCS091|
|Syed Sufyan Ahmed| 18BCS103|
|Shubham Shinde| 18BCS095|


## Project Description

The project focuses on the automated detection of comic mischief in videos, which refers to mild harm inflicted on video characters in a humorous manner. The content may span multiple modalities including video frames, soundtrack, and dialogue. The primary objective is to develop multimodal research techniques to label online video content, particularly to protect young viewers from inappropriate material.


### Objectives

- Develop binary and multi-label classification models to detect comic mischief.
- Preprocess video and audio data using traditional and generator-based approaches.
- Evaluate model performance using various architectures like CNN, VGG16, LSTM, and SVM.


### Methodology
- **Dataset:** Utilized the MOCHA Grand Challenge @ ICMI 2021 dataset, comprising 998 scenes from 347 YouTube and IMDB videos.

- **Tasks:**
    - **Task 1:** Binary detection of comic mischief (presence/absence).
    - **Task 2:** Fine-grained detection with labels: mature humor, slapstick humor, gory humor, and sarcasm.

- **Pre-processing:** Includes traditional video processing, generator-based video processing, and audio preprocessing using ffmpeg and moviepy.
- **Models:** Experimented with 7 models including CNN + Dense, VGG16 + Dense, CNN + LSTM + Dense, VGG16 + LSTM + Dense, and an SVM Audio Classifier.


## Results
- Models were evaluated on training and validation accuracies, with results varying based on approach and model type.
- Detailed analysis and comparisons are provided in the report, highlighting challenges with small datasets and potential improvements.


## Conclusion & Future Scope

The project proposes various approaches for predicting comic mischief, with low accuracy due to limited data. Future work could involve using larger datasets, fine-tuning parameters, and exploring ensemble learning with audio and video modalities.


_Note: Refer to the report (Group18_Mini_Project_Draft_1.pdf) for detailed methodology and model implementation steps._


## References
1. A. Rehman and S. B. Belhaouari, “Deep Learning for Video Classification: A Review”. TechRxiv, 16-Aug-2021.
2. W. Samek et al., “Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications,” Proc. IEEE, vol. 109, no. 3, pp. 247–278, 2021.
3. S. Kiranyaz et al., “1D convolutional neural networks and applications: A survey,” Mech. Syst. Signal Process., vol. 151, 2021.
4. N. Minallah et al., “On the performance of fusion based planet-scope and Sentinel-2 data for crop classification using inception inspired deep convolutional neural network,” PLoS One, vol. 15, no. 9 September, 2020.
5. A. U. Rehman and A. Bermak, “Averaging neural network ensembles model for quantification of volatile organic compound,” 2019 15th Int. Wirel. Commun. Mob. Comput. Conf. IWCMC 2019, pp. 848–852, 2019.
6. D. Brezeale, D. J. Cook, and S. Member, "Automatic video Classification: A survey of the literature," IEEE Transactions on Systems, Man, and Cybernetics, Part C.
7. Z. Wu et al., “Deep learning for video classification and captioning,” Front. Multimed. Res., pp. 3–29, 2017.
8. A. Karpathy et al., "Large-Scale Video Classification with Convolutional Neural Networks," 2014 IEEE Conference on Computer Vision and Pattern Recognition, 2014, pp. 1725-1732, doi: 10.1109/CVPR.2014.223.
9. Simonyan, Karen & Zisserman, Andrew. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. ArXiv 1409.1556.
10. Sun, Chen et al. (2019). VideoBERT: A Joint Model for Video and Language Representation Learning. 7463-7472. 10.1109/ICCV.2019.00756.
11. A Flexible CNN Framework for Multi-Label Image Classification Yunchao Wei et al.

## License

This project is for academic purposes only. Redistribution or modification of the report content requires permission from the authors.

