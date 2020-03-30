# EE6934_Project2: G06

## Project Title: Application of neural networks in estimating the synchronicity of dance moves through pose estimation

### Objective / Problem scope:

CG4002 students are required to develop a wireless wearable dance system, where a server prompts three dancers to synchronously execute a specific dance move. One of their tasks is to determine the synchronicity of their movements and send a time estimate in milliseconds. To evaluate this, a video will be taken of the students dancing and the teaching staff must manually evaluate the recorded video and estimate the sync, which is time-consuming and relatively subjective, making it prone to human error.
In more general terms, many forms of art involve a troupe (e.g. an orchestra, or a group of ballerinas), and what differentiates a mediocre troupe from an outstanding one is the ability for its members to synchronise their movements. Measuring the synchronicity of movements between people is an area that has hardly been explored, and it would be plausible to exploit the advances in pose estimation to do so. For this project, we propose to utilize videos obtained from the first round of evaluations in CG4002 with three predetermined dance moves. We would like to train a model that detects the pose of the performers in each video, and measures the synchronicity between the dancers: a many-to-one prediction problem. 

### Background / Literature search:

Pose estimation is a challenging field for deep learning. Training deep networks to recognise poses face the challenges of great inter-subject variation in anatomical proportions, changes to viewpoints, depth perception and occlusions. However, it is a great way to tackle the objective through the succinct representation of the human body in a model, allowing us to simplify the comparison of movements between subjects. Similarly, Recurrent Neural Networks are often used in tasks which have a temporal component, such as video summarization, machine translation, and speech recognition. To our knowledge, there are currently no applications of RNN’s to this typical problem scope, though there are similar studies where video is used to determine a specific activity, or estimate a specific value. 

### Proposed methodology:

We propose the following methodology:
1.	To examine and compare pre-trained models for pose estimation and choose an appropriate model for our task 
2.	Verify that it can accurately determine pose positions in our training data
3.	Analyse the temporal relationships of the estimated poses through the use of RNN’s and LSTM’s. 
4.	Estimate pose synchronicity analysing time difference between extrema of angular acceleration of each limb.

### Challenges:

The challenges we face during our project would be: Firstly, to create a ground truth for our model for training. Secondly, to integrate the temporal nature of dance movements into the pose estimation architecture and thirdly, to extend the model to analyse the aforementioned temporal measure to determine synchronicity. We can also expect some constraints with regards to the scarcity and quality of the video data. The total videos we can obtain from CG4002 is roughly 33 dance move executions, from the evaluation of 11 groups and three moves on the 3rd of April. Furthermore, as classes are currently being conducted online, we can expect that all 11 videos to be of lower quality. 
