This directory provides the scripts to run CMTs and benchmark methods on the Swissmetro dataset.

## Dependencies

All scripts except `tastenet.py` and `mnlint.py` were run with the following dependencies:

python 2.7.17, joblib 0.13.2, numpy 1.16.6, pandas 0.22.0, rpy2 2.8.6, seaborn 0.9.0, scikit-learn 0.20.3, tensorflow 2.1.0

Note that other R dependencies are specified along with the virtual environment of the source code. The scripts `tastenet.py` and `mnlint.py` were run with the following dependencies:

python 3.8.13, pandas 1.4.2, numpy 1.19.2, tensorflow 2.4.0, tensorboard 2.4.1, tf-nightly 2.5.0, tf-estimator-nightly 2.4.0, keras-preprocessing 1.1.2, keras-tuner 1.1.0

## Description of scripts

`cmt.py` and `cmt+.py`: estimation of the CMT on the 10 data splits for a maximum depth of 14.

`mnlkm.py` and `mnlkm+.py`: estimation of the MNLKM benchmark on the 10 data splits with a search over the number of clusters 5, 10, ... 295. 

`mnldt.py` and `mnldt+.py`: estimation of the MNLDT benchmark on the 10 data splits for a maximum depth of 14. 

`mnlicot.py` and `mnlicot+.py`: estimation of the MNLICOT benchmark on the 10 data splits. The ICOT tree is hardcoded. Code that generates the ICOT tree is provided in the folder `src/ICOT`. The same tree is produced across 5 distinct splits of the data.

`tastenet.py`: estimation of the TasteNet benchmark on the 10 data splits. Code is implemented using the tensorflow library. Input data is used in long format. Note: random seed was not saved.

`mnlint.py`: estimation of the MNLINT benchmark on the 10 data splits. Code is implemented using the tensorflow library. Input data is in used in long format. Note: random seed was not saved.

`plots.py`: visualisation of the Pareto curve showing the predictive performance as a function of the number of segments for MNLKM and CMT.

`prepare_data.py`: code to generate the 10 random splits 75%-12.5%-12.5%, both in long/wide formats. Note: random seed was not saved.




