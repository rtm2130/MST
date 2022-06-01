using DataFrames, MLDataUtils
using Clustering, Distances
using CSV
using Random
using Logging
using NPZ

# Set up Logging - we recommend to use this command to avoid package warnings during the model training process.
logger = Logging.SimpleLogger(stderr,Logging.Warn);
global_logger(logger);

#### Set parameters for the learners
cr = :dunnindex
method = "ICOT_local"
warm_start = :none;
geom_search = true
threshold = 0.99
seed = 1
gridsearch = false
num_tree_restarts = 100
complexity_c = 0.0
min_bucket = 10
maxdepth = 10

###### Step 1: Prepare the data
# Read the data - recommend the use of the (deprecated) readtable() command to avoid potential version conflicts with the CSV package.
# data = readtable("../data/ruspini.csv");

data = npzread("../data/X3.npy")
# data = data[1:200,:]
true_labels = npzread("../data/Y3.npy")
# true_labels = true_labels[1:200,:]

# Convert the dataset to a matrix
data = convert(Matrix{Float64}, data);
true_labels = convert(Matrix{Float64},reshape(true_labels,:,1));

data = DataFrame(hcat(data,true_labels));
data_array = convert(Matrix{Float64}, data);

# Get the number of observations and features
n, p = size(data_array)
data_t = data_array';

##### Step 2: Fit K-means clustering on the dataset to generate a warm-start for ICOT
#Fix the seed
Random.seed!(seed);

# The ruspini dataset has pre-defined clusters, which we will use to select the cluster count (K) for the K-means algorithm.
# In an unsupervised setting (with no prior-known K), the number of clusters for K means can be selected using the elbow method.
K = length(unique(data_array[:,end]))

# Run k-means and save the assignments
kmeans_result = kmeans(data_t, K);
assignment = kmeans_result.assignments;

data_full = DataFrame(hcat(data, assignment, makeunique=true));
names!(data_full, vcat([Symbol(string("x",k)) for k in range(1,p-1)],[:true_labels, :kmean_assign]));

# Prepare data for ICOT: features are stored in the matrix X, and the warm-start labels are stored in y
X = data_full[:,1:(p-1)]; y = data_full[:,:true_labels];

# ##### Step 3a. Before running ICOT, start by testing the IAI license
# lnr_oct = ICOT.IAI.OptimalTreeClassifier(localsearch = false, max_depth = maxdepth,
# 													 minbucket = min_bucket,
# 													 criterion = :misclassification
# 													 )
# grid = ICOT.IAI.GridSearch(lnr_oct)
# ICOT.IAI.fit!(grid, X, y)
# ICOT.IAI.showinbrowser(grid.lnr)

##### Step 3b. Run ICOT

# Run ICOT with no warm-start:
warm_start= :none
lnr_ws_none = ICOT.InterpretableCluster(ls_num_tree_restarts = num_tree_restarts, ls_random_seed = seed, cp = complexity_c, max_depth = maxdepth,
	minbucket = min_bucket, criterion = cr, ls_warmstart_criterion = cr, kmeans_warmstart = warm_start,
	geom_search = geom_search, geom_threshold = threshold);
run_time_icot_ls_none = @elapsed ICOT.fit!(lnr_ws_none, X, y);

ICOT.showinbrowser(lnr_ws_none)

score_ws_none = ICOT.score(lnr_ws_none, X, y, criterion=:dunnindex);
score_al_ws_none = ICOT.score(lnr_ws_none, X, y, criterion=:silhouette);

# # Run ICOT with an OCT warm-start: fit an OCT as a supervised learning problem with labels "y" and use this as the warm-start
# warm_start= :oct
# lnr_ws_oct = ICOT.InterpretableCluster(ls_num_tree_restarts = num_tree_restarts, ls_random_seed = seed, cp = complexity_c, max_depth = maxdepth,
# 	minbucket = min_bucket, criterion = cr, ls_warmstart_criterion = cr, kmeans_warmstart = warm_start,
# 	geom_search = geom_search, geom_threshold = threshold);
# run_time_icot_ls_oct = @elapsed ICOT.fit!(lnr_ws_oct, X, y);
#
# score_ws_oct = ICOT.score(lnr_ws_oct, X, y, criterion=:dunnindex);
# score_al_ws_oct = ICOT.score(lnr_ws_oct, X, y, criterion=:silhouette);
# ICOT.showinbrowser(lnr_ws_oct)

##### Comments
# Note that in this example, the OCT tree and ICOT (OCT warm-start, or no warm-start) result in the same solution.
# This is not generally the case, but can occur when the data is easily separated (as in the ruspini dataset)

# The score printed in the browser view is negative to reflect the formulation as a minimization problem.
# The positive score returned by the ICOT.score() functions reflect the "correct" interpretation of the score,
# in which we seek to maximize the criterion (with a maximum value of 1).

# For larger datasets, we recommend setting warm_start = :oct and threshold = 0.99 to improve the solve time.
