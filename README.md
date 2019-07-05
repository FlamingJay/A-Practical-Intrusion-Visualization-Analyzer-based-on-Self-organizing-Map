# A-Practical-Intrusion-Visualization-Analyzer-based-on-Self-organizing-Map
A Practical Intrusion Visualization Analyzer  based on Self-organizing Map

The library of Self Organizing Map is achieved by https://github.com/JustGlowing/minisom

The whole process includes select a subset of features. The features selected are the same as those in

[1] Modinat M, Abimbola A, Abdullateef B, et al. Gain ratio and decision tree classifier for intrusion detection[J]. International Journal of Computer Applications, 2015, 126(1): 56-59.
[2] Moustafa N, Slay J. A hybrid feature selection for network intrusion detection systems: Central points[J]. arXiv preprint arXiv:1707.05505, 2017.

After that, we apply Isolation Forest to remove the outlier in each class.

We pick out 10% training data to train the model. And we use the entire testing dataset for detection.
![image](https://github.com/FlamingJay/A-Practical-Intrusion-Visualization-Analyzer-based-on-Self-organizing-Map/blob/master/figure/SOM_sample_2class_300W.png)
