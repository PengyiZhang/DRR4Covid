# DRR4Covid: Learning Automated COVID-19 Infection Segmentation from Digitally Reconstructed Radiographs

## Abstract

Automated infection measurement and COVID-19 diagnosis based on Chest X-ray (CXR) imaging is important for faster examination, where infection segmentation is an essential step for assessment and quantification. However, due to the heterogeneity of X-ray imaging and the difficulty of annotating infected regions precisely, learning automated infection segmentation on CXRs remains a challenging task. We propose a novel approach, called DRR4Covid, to learn COVID-19 infection segmentation on CXRs from digitally reconstructed radiographs (DRRs). DRR4Covid comprises of an infection-aware DRR generator, a segmentation network, and a domain adaptation module. Given a labeled Computed Tomography scan, the infection-aware DRR generator can produce infection-aware DRRs with pixel-level annotations of infected regions for training the segmentation network. The domain adaptation module is designed to make the segmentation network trained on DRRs can generalize to CXRs. The statistical analyses made on experiment results have confirmed that our infection-aware DRRs are significantly better than standard DRRs in learning COVID-19 infection segmentation (p < 0.05) and the domain adaptation module can improve the segmentation performance of our network on CXRs significantly (p < 0.05). Without using any annotations of CXRs, our network has achieved a classification score of (Accuracy: 0.949, AUC: 0.987, F1-score: 0.947) and a segmentation score of (Accuracy: 0.956, AUC: 0.980, F1-score: 0.955) on a test set with 558 normal cases and 558 positive cases. Besides, by adjusting the strength of radiological signs of COVID-19 infection in infection-aware DRRs, we estimate the detection limit of X-ray imaging in detecting COVID-19 infection. The estimated detection limit, measured by the percent volume of the lungs that is infected by COVID-19, is 19.43%±16.29%,. Our codes are made publicly available at https://github.com/PengyiZhang/DRR4Covid.


## [InfectionAwareDRRGenerator](/InfectionAwareDRRGenerator)

![](/figs/figuresv2.0_infection_aware.png)


## [FCNClsSegModel](/FCNClsSegModel)

![](/figs/figuresv2.0_fcn.png)



-------------

If you find it useful, please cite:

    @inproceedings{Zhang2020DRR4CovidLA,
    title={DRR4Covid: Learning Automated COVID-19 Infection Segmentation from Digitally Reconstructed Radiographs},
    author={Pengyi Zhang and Yunxin Zhong and Yulin Deng and Xiaoying Tang and Xiaoqiong Li},
    year={2020}
    }

-------------

