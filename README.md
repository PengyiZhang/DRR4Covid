# DRR4Covid: Learning Automated COVID-19 Infection Segmentation from Digitally Reconstructed Radiographs

## Abstract

Automated infection measurement and COVID-19 diagnosis based on Chest X-ray (CXR) imaging is important for faster examination. However, due to the heterogeneous nature of CXRs and the difficulty of precisely annotating, most previous studies have developed classification models rather than segmentation models for COVID-19 diagnosis, and taken the advantages of the interpretability of classification model (e.g., saliency map) to locate the infected regions in lungs roughly. To address this problem, we propose a novel approach, called DRR4Covid, to learn automated COVID-19 diagnosis and infection segmentation on CXRs from digitally reconstructed radiographs (DRRs). Specifically, we design DRR4Covid with a modular framework, which comprises of an infection-aware DRR generator, a classification and/or segmentation network, and a domain adaptation module. The infection-aware DRR generator is able to produce DRRs with adjustable strength of radiological signs of COVID-19 infection, and generate pixel-level infection annotations that match the DRRs precisely, thus enabling the segmentation networks to be trained directly for automated infection segmentation. Although the synthetic DRRs are photo-realistic, there is still a gap between synthetic DRRs and real CXRs, which may lead to a poor model performance on real CXRs. The domain adaptation module is introduced to solve this problem by training networks on unlabeled real CXRs and labeled synthetic DRRs together. Due to the modular framework, our DRR4Covid can be assembled flexibly with off-the-shelf classification and segmentation networks, and domain adaptation algorithms. In this paper, we provide a simple but effective implementation of DRR4Covid by using a domain adaptation module based on Maximum Mean Discrepancy (MMD), and a FCN-based network with a classification header and a segmentation header. Extensive experiment results have confirmed the efficacy of our method; specifically, quantifying the performance by accuracy, AUC and F1-score, our network without using any annotations from CXRs has achieved a classification score of (0.954, 0.989, 0.953) and a segmentation score of (0.957, 0.981, 0.956) on a test set with 794 normal cases and 794 positive cases. Besides, we estimate the sensitive of X-ray images in detecting COVID-19 infection by adjusting the strength of radiological signs of COVID-19 infection in synthetic DRRs. The estimated detection limit of the proportion of infected voxels in the lungs is 19.43%±16.29%, and the estimated lower bound of the contribution rate of infected voxels is 20.0% for significant radiological signs of COVID-19 infection. Our codes will be made publicly available at https://github.com/PengyiZhang/DRR4Covid.





-------------

If you find it useful, please cite:

    @inproceedings{Zhang2020DRR4CovidLA,
    title={DRR4Covid: Learning Automated COVID-19 Infection Segmentation from Digitally Reconstructed Radiographs},
    author={Pengyi Zhang and Yunxin Zhong and Yulin Deng and Xiaoying Tang and Xiaoqiong Li},
    year={2020}
    }

-------------

