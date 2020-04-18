# EE6934_Project2: G06

## Project Title: Application of neural networks in estimating the synchronicity of dance moves through pose estimation

### Objective / Problem scope:

CG4002 students are required to develop a wireless wearable dance system, where a server prompts three dancers to synchronously execute a specific dance move. One of their tasks is to determine the synchronicity of their movements and send a time estimate in milliseconds. To evaluate this, a video will be taken of the students dancing and the teaching staff must manually evaluate the recorded video and estimate the sync, which is time-consuming and relatively subjective, making it prone to human error.
In more general terms, many forms of art involve a troupe (e.g. an orchestra, or a group of ballerinas), and what differentiates a mediocre troupe from an outstanding one is the ability for its members to synchronise their movements. Measuring the synchronicity of movements between people is an area that has hardly been explored, and it would be plausible to exploit the advances in pose estimation to do so. For this project, we propose to utilize videos obtained from the first round of evaluations in CG4002 with three predetermined dance moves. We would like to train a model that detects the pose of the performers in each video, and measures the synchronicity between the dancers: a many-to-one prediction problem. 

### Proposed methodology:

We propose the following methodology:
1.	To examine and compare pre-trained models for pose estimation and choose an appropriate model for our task 
2.	Verify that it can accurately determine pose positions in our training data
3.	Analyse the temporal relationships of the estimated poses through the use of RNN’s and LSTM’s. 
4.	Estimate pose synchronicity analysing time difference between extrema of angular acceleration of each limb.

The workflow is visualized below and will occur in 3 stages.

![workflow img](/images/workflow.png)

Stage 0 (Blue): Preperation of training data by sourcing for videos and manual trimming
Stage 1 (Orange): Extraction of features from videos through pose estimation
Stage 2 (Green): Training of LSTM network for synchronicity estimates

----------------------------

## Running the code
Run the code in the order of stage 0 - 3. Each notebook should be run in order of how the notebook is presented.

----------------------------

### Stage 0: Data preparation
Download videos and introduction on running openpose along with its features. Lastly, explanation on how to estimate synchronicty between extrema from keypoints from openpose.

#### 0 - Video Download 
Download videos from youtube into the `videos` folder. Choose an appropriate youtube video with 3 dancers and copy the appropriate ID (e.g. https:/www.youtube.com/watch?v=AXszbHehGB8 where the ID is AXsbHehGB8). Videos downloaded for this project can be found in `videos`. 

#### 0.1 - Openpose Estimation
Set up the openpose repository for pytorch. This notebook shows how to generate pose estimation for a single image, and includes an example of how openpose is run on a whole video.The output of this stage cuts each video into frames and generates a list of coordinates and scores for each keypoint detected through openpose. The examples generated in this notebook is found under `videos/examples`.

#### 0.2 - Openpose Output Analysis
Display the outputs of openpose. Each frame yields candidate (x, y, score and id) and subset (score and id) for each keypoint.

#### 0.3 - Pose Synchronicity Estimation
Show example synchronicity estimation from peak estimation between the movements of extrema. The per limb estimation of synchronocity is shown. 

----------------------------

### Stage 1: Openpose feature extraction  
Preparation of training data. Features and frames are extracted for 20 frames of each video. 

#### 1 - Frame Generation and KP Feature Extraction: 
Run to divide each video into frames and run openpose on each frame. The features are saved into a npz file.

#### 1.1 - Generate synthetic data
Generate synthetic data by randomly replicating a random subject for a given video. Replications are given random positions with a set delay.

#### 1.a - Dancer Generation
Dancers generated from openpose may be entangled during labling. Dancers are untangled in this process. Outputs of this process is saved in json files under `sync_measure_data`.

#### 1.a1 - Extrema-based features extraction
Extract extrema based features from all subjects. Outputs of this process is saved in json files under `sync_measure_data`.

----------------------------

### Stage 2: Training the LSTM model
Here we train the LSTM model using the the data generated in stage 1. Here we display the notebooks train on real data only. We show several variants based on our optimization process to achieve better results. We provide choices in whether synthetic data and/or real data should be use to train the network. Either a regression model should be run, or classification model. For the regression model, the choice of loss can be selected between L1 and MSE loss. Key functions are included in `vidutils.py` and `trainutils.py`. The final weights from the model are saved in `models`.

#### 2 - RNN Training with KPF
Here the keypoint features are entered into an LSTM network and used to predict synchronicity estimates. The final loss and accuracy are visualized.

#### 2.1 - RNN Training with Normalized KPF
Here each keypoint feature is normalized over the training set before being entered into an LSTM network and used to predict synchronicity estimates. The final loss and accuracy are visualized.

#### 2.2 - RNN Training with Normalized KPF and Penalty
Here each keypoint feature is normalized over the training set before being entered into an LSTM network and used to predict synchronicity estimates. An additional penalty function is included to encourage the predictions to be close to each other (e.g. choosing the class of delay of 100ms when the true delay is 300ms will be penalised more heavily than choosing a delay of 200ms). The final loss and accuracy are visualized.

#### 2.a - RNN Training with EBF
Here, extrema based features are used instead of keypoint features. The final loss and accuracy are visualized.

----------------------------

### Stage 3: Analysing results of model
The models generated from stage 2 are analysed in order to visualize the accuracy and the spread of true values versus predictions. This is done for both the training set and test set. 

#### 3 - Results for RNN with KPF
Load weights and visualize accuracies and distribution of results for final model in stage 2.

#### 3.1 - Results for RNN with Normalized KPF
Load weights and visualize accuracies and distribution of results for final model in stage 2.1.

#### 3.2 - Results for RNN with Normalized KPF and Penalty
Load weights and visualize accuracies and distribution of results for final model in stage 2.2.

#### 3a - Results for RNN with EBF
Load weights and visualize accuracies and distribution of results for final model in stage 2a.

----------------------------
