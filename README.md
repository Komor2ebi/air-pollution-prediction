[![Shipping files](https://github.com/neuefische/ds-ml-project-template/actions/workflows/workflow-02.yml/badge.svg?branch=main&event=workflow_dispatch)](https://github.com/neuefische/ds-ml-project-template/actions/workflows/workflow-02.yml)

# ML Project Description

**Project** 
The <u>Air Pollution Challenge on Zindi</u> is focused on predicting air quality in various cities around the globe using satellite data. Participants in this challenge are tasked with using machine learning techniques to estimate pollution levels, specifically PM2.5 concentrations. The challenge leverages data that spans across different geographical locations and conditions, providing a platform for developers to test and enhance their predictive modeling skills in real-world scenarios.

The competition is designed to address the growing issue of urban air pollution, providing insights and solutions that can help in managing air quality in densely populated cities. The use of satellite data in this context allows for a wide-reaching analysis that can cover areas where ground-based sensors might not be available, thus filling crucial data gaps in global environmental health studies.

For more detailed discussions and resources related to this challenge, including data sets and participant discussions, you can visit the official challenge page on Zindi's website: Urban Air Pollution Challenge on Zindi.

![Our Mission](/images/overview_air_pollution_challenge.png)


**Team**
Our team consists of three scientist, Iris Winkler, Carlos Duque and Johannes Gooth. Together we explored, cleaned, analyzed the data and exploited various machine learning algorithms to predict PM2.5 concentrations according to the satellite data provided by Zindi.  The Zindi challenge was already closed at that time, thus we did not submit our achievements. To structure our workflow we made use of Git Projects. We finally presented our findings and uploaded the presentation as pdf to this repository.


## Set up your Environment

### **`macOS`** type the following commands : 

- For installing the virtual environment you can either use the [Makefile](Makefile) and run `make setup` or install it manually with the following commands:

     ```BASH
    make setup
    ```
    After that active your environment by following commands:
    ```BASH
    source .venv/bin/activate
    ```
Or ....
- Install the virtual environment and the required packages by following commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    
### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-bash` CLI :
  
    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **`Note:`**
    If you encounter an error when trying to run `pip install --upgrade pip`, try using the following command:
    ```Bash
    python.exe -m pip install --upgrade pip
    ```


   
## Usage

In order to train the model and store test data in the data folder and the model in models run:

**`Note`**: Make sure your environment is activated.

```bash
python example_files/train.py  
```

In order to test that predict works on a test set you created run:

```bash
python example_files/predict.py models/linear_regression_model.sav data/X_test.csv data/y_test.csv
```

## Limitations

Development libraries are part of the production environment, normally these would be separate as the production code should be as slim as possible.


