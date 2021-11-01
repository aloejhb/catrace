# CaTrace
A python package for calcium imaging time trace analysis

## Usage
  See demos/ for how to use the code.\
  analysis_multi_dataset.ipynb shows how to do multiple dataset analysis.\
  analysis_OB.py shows how to do single dataset analysis.
## Structure of the package
  * dataio: data reading and writing, contains functions to read data from MATLAB output.
  * process_time_trace: preprocess time trace for further analysis.
  * manifold_embed: functions for dimensionality reduction.
  * pattern_correlation: compute pattern correlation and plotting.
  * plot_trace: plotting functions to visualize calcium traces.
  * exp_collection: batch processing for multiple datasets.
