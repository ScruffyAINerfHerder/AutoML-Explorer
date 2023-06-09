AutoML Explorer App

This application provides a user-friendly interface for users to upload, profile and analyze datasets using machine learning. Built with Python and Streamlit, this app includes profiling capability using pandas_profiling and AutoML functionalities with the PyCaret library.

Running the AutoML Explorer App
To build the docker image, right click on the Dockerfile and select Build Image

Once the image is built to run type 'docker run -p 8501:8501 automl_project' into CMD (in the Dockerfile you can replace the port# with any you like for Streamlit)

Go to the browser and type localhost:8501 (or whatever port number you like)

Navigation
The app provides four main functionalities, which can be accessed through the sidebar:

Upload: Allows you to upload your dataset for modelling. You can upload your dataset by choosing this option and then using the 'Upload Dataset Here' button to upload a .csv file.

Profiling: Displays an automated data explorer report generated using pandas profiling. After you've uploaded your data, you can choose this option to get a detailed report about your data.

AutoML: Allows you to perform Automated Machine Learning (AutoML) on your uploaded dataset. You need to select your target variable from a dropdown menu that contains all the columns from your dataset. After selecting the target variable, press the 'Train Model' button to start the training. It will display the settings used for training and a summary of the best model found.

Download: Allows you to download the trained model. Once the model has been trained using AutoML, you can download the trained model in .pkl format using the 'Download the Model' button. You can then open and run this model using a Jupyter notebook.

Sidebar
The sidebar of the app provides some additional features. It displays the title of the app and a nice image. It also contains an information box explaining what the app does. Lastly, it allows you to navigate between the different functionalities of the app by choosing from the radio buttons.

Dataset
The dataset used for this application should be in .csv format and should be uploaded through the 'Upload' function. Once uploaded, the data is stored in a file named 'sourcedata.csv'.

Profiling
This application uses pandas profiling to generate a detailed report about the uploaded dataset. This report includes information about the variables in the dataset, interactions between variables, missing values, and much more.

AutoML
The application uses the PyCaret library to perform Automated Machine Learning (AutoML). This process includes setting up the environment for the ML task, comparing different models to find the best one, displaying the results of the best model, and saving the model.

Download
Once the best model is found using AutoML, it is saved in a file named 'best_model.pkl'. This model can be downloaded by choosing the 'Download' option.

In conclusion, this application provides a user-friendly interface to perform Automated Machine Learning on your datasets. It is easy to use and provides detailed information about your data and the models trained on it.






#   A u t o M L - E x p l o r e r  
 