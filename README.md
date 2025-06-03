<div align="center">
<h1>FACEGroup: Feasible and Actionable Counterfactual Explanations for Group Fairness Auditing</h1>

**Christos Fragkathoulas<sup>1,2</sup>, Vassiliki Papanikou<sup>1,2</sup>, Evaggelia Pitoura<sup>1, 2</sup>, Evimaria Terzi<sup>2,3</sup>**

<sup>1</sup>University of Ioannina,
<sup>2</sup>Archimedes, Athena Research Center, Greece,
<sup>3</sup>Boston University, USA

[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

</div>

This repository contains the official implementation of the paper: **FACEGroup: Feasible and Actionable Counterfactual Explanations for Group Fairness**

## Contents

1.  [Abstract](#Abstract)
2.  [Installation](#installation)
3.  [Acknowledgement](#Acknowledgement)

## Abstract
This paper introduces the first graph-based framework for generating group counterfactual explanations
to audit model fairness, a key aspect of trustworthy machine learning. Counterfactual explanations help assess unfairness by revealing how inputs must change to achieve a desired outcome. Our framework, FACEGroup (Feasible and Actionable Counterfactual Explanations for Group Fairness Auditing), models real-world feasibility constraints, constructs subgroups with similar counterfactuals, and addresses key trade-offs in counterfactual generation, distinguishing it from existing methods. To evaluate fairness, we introduce metrics tailored to group counterfactuals generation trade-offs.
Experiments on benchmark datasets show that FACEGroup effectively generates feasible group counterfactuals while accounting for trade-offs, and that our metrics capture and quantify fairness disparities.

## Installation

1.  Clone the repository:
      ```bash
      git clone [https://github.com/xristosfrag/FACEGroup-Feasible-and-Actionable-Counterfactual-Explanations-for-Group-Fairness-Auditing](https://github.com/xristosfrag/FACEGroup-Feasible-and-Actionable-Counterfactual-Explanations-for-Group-Fairness-Auditing)
      ```

2. Navigate to the project directory and then create a virtual environment (optional but recommended):
    ```bash
    pip install virtualenv
    python<version> -m venv <virtual-environment-name>
    source <virtual-environment-name>/bin/activate  # On Windows, use `<virtual-environment-name>\Scripts\activate`
     ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
     ```

## Acknowledgement
This work has been partially supported by project MIS 5154714 of the National Recovery and Resilience Plan Greece 2.0 funded by the European Union under the NextGenerationEU Program.