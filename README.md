Code implementation for *Exploiting Labeled and Structured Data for Fine-Grained Domain Adaptation to Unseen Target Classes*.
## Train 
Scripts to run the training are in `PSZS/Scripts`. 
The `run.py` script assumes working with a (set of) source and target domain(s).
The `runSource.py` script can be used to train a model on only source domain data. For the source only training domain adaptation is not supported i.e. only `--method erm` works.
### Datasets
Currently the following datasets are directly supported:
- CompCars
    - CompCarsModel
    - CompCarsHierarchy
- CUB-Paintings
    - CUBSpecies
    - CUBHierarchy
These have to be specified as the `--data` argument. If a hierarchical dataset is specified but the `head-type` is non hierarchical the dataset will be automatically changed to the non hierarchical version. 
Whether a dataset is hierarchical can be specified via the `multi_label` attribute of the parent class `CustomDataset` (albeit it currently is hard coded in the run scripts).\
New datasets can be added as inherting from `CustomDataset` and should come with a dataset descriptor and annotation file.\
For hierarchical datasets the level structure is assumed to be from coarse to fine.
## Supported DA 
Currently implemented and supported methods that can be specified via the `--method` parameter are:
- DANN
- ADDA
- JAN
- CCSA
- MCC
- MCD
- MDD
- PAN
- UJDA
Additional/Custom specification of feature or logit loss can be given via `--model-kwargs` as `feature_loss_func` and `logit_loss_func` alongside the specified accumulation scheme and other parameters.\
For details see the base class for all models `PSZS/Models/CustomModel`.
## Classifiers
The type of classifier to be used is specified via `--classification-type` with available types:
- DefaultClassifier
- MaskedClassifier
- SeparatedClassifier
The type of classification head used is specified via `--head-type` and needs to match the used dataset type.
If not specified, a SimpleHead without hierarchy is used.\
All options are found under `PSZS/Classifiers`.
## Evaluation
For evaluation the following options are available:
- Create a confusion matrix as an `.html` using the pycm library
- Create an excel report that tracks which classes are misclassified most commonly and what are the confusions
- Create an excel in addition to the .csv file with the final metrics
- Create a t-SNE visualization of the features
- Make a p-value significance test
- Measure A-Distance (WIP/currently buggy)
The first three options are controllable via `--create-excel`, `--create-class-summary` and `--create-report` while the other options have no direct implementation but the code that achieves it is under `PSZS/Utils/Eval`.