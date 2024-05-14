<h1>INTEGRATING OUR MODEL WITH HPCC - PHASE 1</h1>

This documentation describes our process in which we put up our entire model on ECL which is able to bring out the medical parameters of an input string, and have created a frontend interface using Java SpringBoot and Thymeleaf as a template engine for development. A POST request was created onto the roxie server of HPCC Systems. The data on output is being displayed as shown below:

![image](https://github.com/PRASHANT-tech870/NLP-pipeline-for-EHR/assets/153075137/8237f1f4-0bef-429b-a39f-d6adbf62200f)

The following is done in this repository:
Two important classes have been created for this purpose: model.java and MyController.java. The former consists of the model parameters to be entered, in our case, the string by the user. The latter class is used for creating the GET request, using index method, which displays index.html(in resources/templates) as well as a post request using sendJsonQuery method, which displays request.html. 
The following is how response looks like:
![image](https://github.com/PRASHANT-tech870/NLP-pipeline-for-EHR/assets/153075137/9305af64-f65d-42aa-86d7-425d80d006c1)
