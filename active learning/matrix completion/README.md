An implementation of matrix completion using Singular Value Decomposition (SVD).
We used the power method of SVD in so that we can account for sparse matrices;
we use a sparse matrix due to the size of the dataset.

We are currently using the Movie Lens dataset (included in directory 'dataset')
which will need to be reconfigured for the dataset for this project.

Items to be Done
1) Fix SVD bug
    Values appear to be off, but correlated to actual value
    Normalization may help this, as this is a problem due to the sparse matrix
        The current algorithm worked with a dense matrix, but that is not
        possible with our dataset size.
2) Refactor svd_model.py code
    Should have a clearer method of training/testing/cross validation
3) Reconfigure for Joke Dataset
    Can only be done after joke dataset is set, should only mean changing parts
        in data_loader.py

