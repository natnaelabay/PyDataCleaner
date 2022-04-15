# PyDataCleaner Docs

pyDataCleaner is a simple utility tool that helps automate the data cleaning process for data modeling

## commonly used data cleaning steps

![image pipeline](./images/pipeline.png)


## Features provided by this mini

1. Load a dataset with any of these types
    - CSV, JSON, XLSX, XLS, DAT
2. Detect missing values and apply necessary imputation techniques
3. Detect outliers and apply the necessary removal techniques
4. Detect Categorical features and apply encoding techniques 
5. Feature transformation and selection techniques to create new features and choose a specific set of variables that best represent the data characteristics


File Structure 

```
project
│   README.md
│
└───pyDataCleaner
│   │   __init__.py
│   │   cleaner.txt
│   │
│   └───components
│       │   feature_selector.py
│       │   file112.txt
```

There are only a few classes in this modules most people will only be working with the `AutoCleaner` class, which as the name suggests automates all the tiresome work and gives you an elegant API to work with. <br>

### Example 1
<b>Remark</b> `Lets say there is a dataset "datasets/test.csv"`

1. loading the dataset
    Before you start using all the features you are going to need to create an AutoCleaner object
    ```
    # create an AutoCleaner Object
    cleaner = AutoCleaner(
        path_to_the_dataset
    )
    ```
