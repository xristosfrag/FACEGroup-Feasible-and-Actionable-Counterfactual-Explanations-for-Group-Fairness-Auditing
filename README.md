# FACEGroup: Feasible and Actionable Counterfactual Explanations for Group Fairness Auditing
## Abstract

This paper introduces the first graph-based framework for generating group counterfactual explanations
to audit model fairness, a key aspect of trustworthy machine learning. Counterfactual explanations help assess unfairness by revealing how inputs must change to achieve a desired outcome. Our framework, FACEGroup (Feasible and Actionable Counterfactual Explanations for Group Fairness Auditing), models real-world feasibility constraints, constructs subgroups with similar counterfactuals, and addresses key trade-offs in counterfactual generation, distinguishing it from existing methods. To evaluate fairness, we introduce metrics tailored to group counterfactuals generation trade-offs.
Experiments on benchmark datasets show that FACEGroup effectively generates feasible group counterfactuals while accounting for trade-offs, and our metrics capture and quantify fairness disparities.

## Installation

To install the required dependencies for running the FACEGroup framework, follow these steps:

1. **Navigate to the project directory and then create a virtual environment (optional but recommended)**:
   ```bash
    pip install virtualenv
    python<version> -m venv <virtual-environment-name>
    source <virtual-environment-name>/bin/activate  # On Windows, use `<virtual-environment-name>\Scripts\activate`

2. **Install the required packages**:
   ```bash
    pip install -r requirements.txt