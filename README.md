# MTCNet
Implementation of "MTCNet: Multitask consistency network with single temporal supervision for semi-supervised building change detection"

Building change detection is crucial for urban development. Over the past few years, deep learning based change
detection researches have achieved impressive progress. However, the limitation is that a large number of change
labels are required. Semi-supervised change detection requires only a small number of change labels and is
receiving increasing attention. It is true that labeling building footprint in a single temporal image is low-cost
compared with labeling changes, which requires constantly comparing bi-temporal images. Meanwhile, the
building ground truth (especially in pre-temporal phase) is more easily available. Therefore, single temporal
building priori is used as supervision signals to improve semi-supervised change detection performance. In this
paper, a multitask consistency network (MTCNet) with single temporal supervision is proposed, using a small
number of change labels and single temporal building labels for semi-supervised building change detection. To
make full advantage of the building prior information, the multitask learning strategy is adopted which performs
both building segmentation and change detection tasks to obtain discriminative features. To exploit unlabeled
data, a task-level consistency learning strategy is proposed to enhance the generalization ability. Experiments on
two building change detection datasets validate the effectiveness of our method. It is found that using only 10%
change labels and the corresponding single temporal building labels in Guangzhou dataset, MTCNet improves the
F1-score by 21.04% compared to the supervised single change detection task method and improves by more than
9.95% compared to other semi-supervised change detection methods. Moreover, if extra T1 labels are provided,
the F1-score can be further improved by 7.14%.

<img width="977" alt="image" src="https://github.com/user-attachments/assets/94a74a88-4951-4a46-9a8e-716c09f89c64">

<div align=center>
	<img width="756" alt="image" src="https://github.com/user-attachments/assets/dacc940d-52a4-4352-a2a8-4cd99c0f3a7f">
</div>


