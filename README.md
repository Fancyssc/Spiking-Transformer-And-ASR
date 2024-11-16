Here is the repo to repoduce Spiking Transformer to complete the final assignment of UCAS Speech Signal Processing 

## dataset used
SHD

## brain-inspire framework
Braincog is used to produced all codes

## results
| Model           | Step | Acc@1 |
|------------------|------|-------|
| QKFormer + TIM   | 32   | 85.02 |
| QKFormer + TIM    | 10   | 80.43 |
|Spikformer | 10 | 85.1 |  
|Spikformer + TIM | 10 | 86.3| 


## referred code repo
(Spikformer(ICLR2023))[https://github.com/ZK-Zhou/spikformer]
(TIM(IJCAI2024))[https://github.com/Fancyssc/TIM]
(QKFormer(Nips2024))[https://github.com/zhouchenlin2096/QKFormer]


## reference
```
[1]	Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning internal representations by error propagation, parallel distributed processing, explorations in the microstructure of cognition, ed. de rumelhart and j. mcclelland. vol. 1. 1986. Biometrika, 71(599-607), 
[2]	Wang, D., Wang, X., & Lv, S. (2019). An overview of end-to-end automatic speech recognition. Symmetry, 11(8), 1018.
[3]	Amodei, D., Ananthanarayanan, S., Anubhai, R., Bai, J., Battenberg, E., Case, C., ... & Zhu, Z. (2016, June). Deep speech 2: End-to-end speech recognition in english and mandarin. In International conference on machine learning (pp. 173-182). PMLR.
[4]	Chan, W., Jaitly, N., Le, Q. V., & Vinyals, O. (2015). Listen, attend and spell. arXiv preprint arXiv:1508.01211.
[5]	Vaswani, A. (2017). Attention is all you need. Advances in Neural Information Processing Systems.
[6]	Wu, J., Yılmaz, E., Zhang, M., Li, H., & Tan, K. C. (2020). Deep spiking neural networks for large vocabulary automatic speech recognition. Frontiers in neuroscience, 14, 199.
[7]	Cramer, B., Stradmann, Y., Schemmel, J., & Zenke, F. (2020). The heidelberg spiking data sets for the systematic evaluation of spiking neural networks. IEEE Transactions on Neural Networks and Learning Systems, 33(7), 2744-2757.
[8]	Dong, L., Xu, S., & Xu, B. (2018, April). Speech-transformer: a no-recurrence sequence-to-sequence model for speech recognition. In 2018 IEEE international conference on acoustics, speech and signal processing (ICASSP) (pp. 5884-5888). IEEE
[9]	Zeyer, A., Bahar, P., Irie, K., Schlüter, R., & Ney, H. (2019, December). A comparison of transformer and lstm encoder decoder models for asr. In 2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU) (pp. 8-15). IEEE.
[10]	Bhangale, K. B., & Kothandaraman, M. (2022). Survey of deep learning paradigms for speech processing. Wireless Personal Communications, 125(2), 1913-1949.
[11]	Collobert, R., Puhrsch, C., & Synnaeve, G. (2016). Wav2letter: an end-to-end convnet-based speech recognition system. arXiv preprint arXiv:1609.03193.
[12]	Radha, K., Bansal, M., & Pachori, R. B. (2024). Speech and speaker recognition using raw waveform modeling for adult and children’s speech: A comprehensive review. Engineering Applications of Artificial Intelligence, 131, 107661.
[13]	Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 33, 12449-12460.
[14]	Li, H., Liu, H., Ji, X., Li, G., & Shi, L. (2017). Cifar10-dvs: an event-stream dataset for object classification. Frontiers in neuroscience, 11, 309.
[15]	Cramer, B., Stradmann, Y., Schemmel, J., & Zenke, F. (2020). The heidelberg spiking data sets for the systematic evaluation of spiking neural networks. IEEE Transactions on Neural Networks and Learning Systems, 33(7), 2744-2757.
[16]	Wu, J., Chua, Y., & Li, H. (2018, July). A biologically plausible speech recognition framework based on spiking neural networks. In 2018 international joint conference on neural networks (IJCNN) (pp. 1-8). IEEE.
[17]	Zhou, Z., Zhu, Y., He, C., Wang, Y., Yan, S., Tian, Y., & Yuan, L. (2022). Spikformer: When spiking neural network meets transformer. arXiv preprint arXiv:2209.15425.
[18]	Shen, S., Zhao, D., Shen, G., & Zeng, Y. (2024). TIM: An Efficient Temporal Interaction Module for Spiking Transformer. arXiv preprint arXiv:2401.11687.
[19]	Zhou, C., Zhang, H., Zhou, Z., Yu, L., Huang, L., Fan, X., ... & Tian, Y. (2024). QKFormer: Hierarchical Spiking Transformer using QK Attention. arXiv preprint arXiv:2403.16552.
```
