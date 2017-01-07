PMML Generator 
==============
The script pmml.py can be used to translate the LightGBM models, found in LightGBM_model.txt, to  predictive model markup language (PMML). These models can then be imported by other analytics applications. The models that the language can describe includes decision trees. The specification of PMML can be found here at the Data Mining Group's [website](http://dmg.org/pmml/v4-3/GeneralStructure.html). 

In order to generate pmml files do the following steps.
```
lightgbm config=train.conf
python pmml.py LightGBM_model.txt
```
The python script will create a file called **LightGBM_pmml.xml**. Inside the file you will find a `MiningModel` tag. In there you will find `TreeModel` tags. Each `TreeModel` tag contains the pmml translation of a decision tree inside the LightGBM_model.txt file. The model described by the **LightGBM_pmml.xml** file can be transferred to other analytics applications. For instance you can use the pmml file as an input to the jpmml-evaluator API. Follow the steps below to run a model described by **LightGBM_pmml.xml**. 

##### Steps to run jpmml-evaluator
1, First clone the repository
```
git clone https://github.com/jpmml/jpmml-evaluator.git
```
2, Build using maven
```
mvn clean install
```
3, Run the EvaluationExample class on the model file using the following command
```
java -cp example-1.3-SNAPSHOT.jar org.jpmml.evaluator.EvaluationExample --model LightGBM_pmml.xml --input input.csv --output output.csv
```
Note, in order to run the model on the input.csv file, the input.csv file must have the same number of columns as specified by the `DataDictionary` field in the pmml file. Also, the column headers inside the input.csv file must be the same as the column names specified by the `MiningSchema` field. Inside output.csv you will find all the columns inside the input.csv file plus a new column. In the new column you will find the scores calculated by processing each rows data on the model. More information about jpmml-evaluator can be found at its  [github repository](https://github.com/jpmml/jpmml-evaluator).