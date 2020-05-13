# AGCN-SIGIR2020
Joint Item Recommendation and Attribute Inference: An Adaptive Graph Convolutional Network Approach 
![](https://github.com/yimutianyang/AGCN/blob/master/figure/framework.png)

In many recommender systems, users and items are associated with attributes, and users show preferences to items. The attribute information describes users’ (items’) characteristics and has a wide range of applications, such as user profiling, item annotation, and feature enhanced recommendation. In practice, the attribute values are often incomplete with many missing attribute values. Therefore, item recommendation and attribute inference have become two main tasks in these platforms. In this paper, we define these two tasks in an attributed user-item bipartite graph, and propose an Adaptive Graph Convolutional Network (AGCN) approach for joint item recommendation and attribute inference. The key idea of AGCN is to iteratively perform two parts: 1) Learning graph embedding parameters with previously learned approximated attribute values to facilitate item recommendation and attribute inference update; 2) Sending the approximated updated attribute values back to the attributed graph for better graph embedding learning. Therefore, AGCN could adaptively adjust the graph embedding learning parameters by incorporating both the given attributes and the estimated attribute values, in order to provide weakly supervised information to refine the two tasks. Extensive experimental results clearly show the effectiveness of the proposed model.

We provide Tensorflow implementations of proposed AGCN model on the Movielens-20M dataset.

Prerequisites
-------------
* Tensorflow 1.9.0
* Python 3.6
* NVIDIA GPU + CUDA CuDNN

Usage
-----
* Dataset:<br>
Under the data folder(cd ./data)
* Item Recommendation Task:<br>
cd ./code<br>
python train_task1.py<br>
python test_task1.py<br>
* Attributes Inference Task:<br>
cd ./code<br>
python task2.py<br>



