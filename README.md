## 1.	Problem Definition:

  This project aims to detect pneumonia disease in individuals using x-ray images by employing deep learning techniques. A Convolutional Neural Network (CNN) model has been developed for this purpose. Early diagnosis of pneumonia is crucial in the treatment process. Therefore, developing an automated detection method can expedite the diagnosis process and improve accuracy.


## 2.	Explanation of the neural network model used:

  The neural network model used in this project is a Convolutional Neural Network (CNN). CNNs consist of structured layers that are particularly effective in analyzing visual data. In this project, a CNN model has been designed with several convolutional layers, pooling layers, and fully connected layers. The convolutional layers use filters to create feature maps on the images, while pooling layers summarize these feature maps. The fully connected layers are responsible for classification. During the learning process, the model optimizes weight and bias values to analyze x-ray images and predict the presence of pneumonia disease.


## 3.	Description of the Dataset and its Acquisition: 

  The dataset used in this project consists of x-ray images of pneumonia patients and healthy individuals. More than 100 images were utilized in total. The dataset was obtained from hospitals and healthcare institutions, comprising real x-ray images. The images were categorized into two classes: normal and pneumonia, with each image labeled accordingly.
  
  The preprocessing of the dataset involved steps such as standardizing image dimensions, adjusting color channels, and applying noise reduction techniques. Additionally, the dataset was split into training, validation, and test sets. The training set was used to learn the weight and bias values of the model, while the validation and test sets were used to evaluate the model's performance and analyze the results.
  
  You can access to dataset via the [link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)


## 4.	Results: 

  The CNN model achieved highly successful results. Evaluations on the test set demonstrated an accuracy rate of over 90% in accurately detecting pneumonia disease. The model's performance was assessed using metrics such as accuracy, recall, and precision. These results indicate that the developed model is effective in pneumonia detection and could potentially serve as a valuable tool in clinical applications.


## 5.	Discussion: 
  While the CNN model used in this project yields successful results for pneumonia detection, certain discussion points arise at this stage of the report. Firstly, increasing the dataset size and utilizing a more diverse set of data could enhance the model's generalization and its ability to recognize different scenarios. Moreover, methods such as hyperparameter tuning and exploring more complex neural network architectures could be considered to further improve the model's accuracy.
  
  Furthermore, discussions can be held regarding the integration of the model into clinical applications, obtaining real-time results, and ensuring the model's reliability. In clinical applications, the model's performance should be examined under the influence of various factors, such as different patient groups and image qualities. Additionally, efforts should be made to address false positive and false negative outcomes of the model and develop improvement strategies.

  In conclusion, this project presents an effective solution for pneumonia detection using deep learning techniques. The developed CNN model exhibits high accuracy rates and can be evaluated for usability in clinical applications. However, further research and testing are required, and therefore, it is recommended to conduct more validation and improvement studies to make the model more practical and reliable.


## 6.	How to Run
To run our code, some Python libraries should be installed: NumPy, Pandas, Matplotlib, Seaborn, TensorFlow, Keras, and OpenCV. After the installation of these libraries, our dataset named “chest_xray” should be on the same folder with our “PneumoniaDetection.py” file. After these steps, just run our Python file.


## Some visualization outputs:
![fig1](https://github.com/metehannguzel/pneumonia-detection-cnn/assets/66705106/57faa993-8978-4d8e-a43b-2e0aac48a9aa)

![fig2](https://github.com/metehannguzel/pneumonia-detection-cnn/assets/66705106/a7346087-57d5-47c1-bc71-7a0556ac612c)

![fig3](https://github.com/metehannguzel/pneumonia-detection-cnn/assets/66705106/4bf50eb1-2fee-411a-a2ed-b3d86abeec6b)

![fig4](https://github.com/metehannguzel/pneumonia-detection-cnn/assets/66705106/202019bf-0253-4531-aa90-d55d66e77d7d)

![fig5](https://github.com/metehannguzel/pneumonia-detection-cnn/assets/66705106/5d9ca0cd-9a63-48ea-a386-32eba39d3e42)

![fig6](https://github.com/metehannguzel/pneumonia-detection-cnn/assets/66705106/5930a4c0-0885-4c92-a709-a825cedb45ed)

![fig7](https://github.com/metehannguzel/pneumonia-detection-cnn/assets/66705106/c9adbce2-815d-4555-a760-32f3a0354cb5)

![fig8](https://github.com/metehannguzel/pneumonia-detection-cnn/assets/66705106/cbd86e41-c150-4d2a-a77b-bf89d174336b)

![fig9](https://github.com/metehannguzel/pneumonia-detection-cnn/assets/66705106/939dec68-308a-460b-bb23-36a7067e94d0)
