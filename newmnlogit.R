library("mlogit")
library("mnlogit")
library("matrixStats")
library("data.table")
library("limSolve")
#library("pracma")
reordering <- function (varList, choices)
{
  if (length(varList) == 0)
    return(NULL)
  K <- length(as.vector(choices))
  p <- length(as.vector(varList$indSpVar))
  f <- length(as.vector(varList$csvChCoeff))
  d <- length(as.vector(varList$csvGenCoeff))
  orig <- c(if (p > 0) rep(1:p, K - 1) else NULL, if (f > 0) rep((p +
                                                                    1):(p + f), K) else NULL, if (d > 0) (p + f + 1):(p +
                                                                                                                        f + d) else NULL)
  order(orig)
}
makeCoeffNames <- function (varNames, choices)
{
  if (length(varNames) == 0)
    return(NULL)
  choices <- as.vector(choices)
  coeffName <- c(outer(varNames$indSpVar, choices[-1], paste,
                       sep = ":"), outer(varNames$csvChCoeff, choices, paste,
                                         sep = ":"), varNames$csvGenCoeff)
}
mnlogit <- function (formula, data, choiceVar = NULL, maxiter = 50, ftol = 1e-06,
                     gtol = 1e-06, weights = NULL, ncores = 1, na.rm = TRUE, print.level = 0,
                     linDepTol = 1e-06, start = NULL, alt.subset = NULL, ...)
{
  startTime <- proc.time()[3]
  initcall <- match.call()
  if (!is.data.frame(data))
    stop("data must be a data.frame in long format or a mlogit.data object")
  if (ncores < 1) {
    ncores <- 1
    warning("Setting ncores equal to: 1")
  }
  if (!is.null(choiceVar) && is.factor(data[[choiceVar]])) {
    warning(paste("Column", choiceVar, "in data will NOT be treated as",
                  "factor, but as character string!"))
  }
  if (is.null(choiceVar) && !any(class(data) == "mlogit.data"))
    stop("Arg data MUST be a mlogit.data object when arg choiceVar = NULL")
  if (is.null(choiceVar)) {
    choiceVar <- "_Alt_Indx_"
    data[[choiceVar]] <- attr(data, "index")$alt
  }
  formula <- parseFormula(formula)
  response <- attr(formula, "response")
  interceptOn <- attr(formula, "Intercept")
  csvChVar <- attr(formula, "csvChCoeff")
  indspVar <- attr(formula, "indSpVar")
  csvGenVar <- attr(formula, "csvGenCoeff")
  covariates <- c(csvChVar, indspVar, csvGenVar)
  varNames <- attr(formula, "varNames")
  if (is.null(covariates) && !interceptOn)
    stop("Error! Predictor variable(s) must be specified")
  if (is.null(response))
    stop("Error! Alternative variable must be specified")
  if (!is.null(alt.subset)) {
    if (sum(unique(data[[choiceVar]]) %in% alt.subset) <
        2)
      stop("Error! Atleast 2 alternatives in data must be in alt.subset")
    keepRows <- data[[choiceVar]] %in% alt.subset
    if (sum(keepRows) <= 0)
      stop("Error! No altrnative in 'alt.subset' is in data.")
    data <- data[keepRows, , drop = FALSE]
  }
  choice.set <- unique(data[[choiceVar]])
  K <- length(choice.set)
  if (nrow(data)%%K)
    stop("Mismatch between number of rows in data and number of choices.")
  N <- nrow(data)/K
  if (!is.null(weights) && length(weights) != N)
    stop("Length of 'weights' arg must match number of observations in data.")
  if (!is.null(weights) && !all(weights > 0))
    stop("All entries in 'weights' must be strictly positive.")
  if (!is.null(weights))
    weights <- weights * N/sum(weights)
  data <- data[, c(varNames, choiceVar)]
  na.rows <- c()
  for (col in 1:dim(data)[2]) na.rows <- union(na.rows, which(is.na(data[[col]])))
  Ndropped <- 0
  if (length(na.rows) > 0) {
    if (!na.rm)
      stop("NA present in input data.frame with na.rm = FALSE.")
    keepRows <- rep(TRUE, nrow(data))
    keepRows[na.rows] <- FALSE
    for (i in 1:N) {
      if (!all(keepRows[((i - 1) * K + 1):(i * K)]))
        keepRows[((i - 1) * K + 1):(i * K)] <- FALSE
    }
    data <- data[keepRows, , drop = FALSE]
    if (!is.null(weights)) {
      weights <- weights[keepRows[seq(1, N * K, K)]]
    }
    N <- nrow(data)/K
    Ndropped <- (length(keepRows) - sum(keepRows))/K
  }
  if (print.level && Ndropped > 0)
    cat(paste("Num of dropped observations (due to NA)  =",
              Ndropped, "\n"))
  #data <- data[order(data[[choiceVar]]), ]
  data = as.data.table(data)
  setkeyv(data, choiceVar)
  data = as.data.frame(data)
  choice.set <- unique(data[[choiceVar]])
  respVec <- data[[attr(formula, "response")]]
  if (is.factor(respVec))
    respVec <- droplevels(respVec)
  respVec <- as.numeric(respVec)
  min.respVec <- min(respVec)
  spread <- max(respVec) - min.respVec
  if (spread != 1) {
    stop(paste("Response variable", attr(formula, "response"),
               "must be a factor with exactly 2 levels."))
  }
  respVec <- respVec - min.respVec
  freq.choices <- colSums(matrix(respVec, nrow = N, ncol = K))/N
  loFreq <- min(freq.choices)
  loChoice <- choice.set[which(loFreq == freq.choices)]
  names(freq.choices) <- choice.set
  #if (loFreq < 1e-07) {
    #cat("Frequencies of alternatives in input data:\n")
    #print(prop.table(freq.choices), digits = 4)
    #stop(paste("Frequency, in response, of choice:", loChoice,
    #           "< 1e-7."))
  #}
  formDesignMat <- function(varVec = NULL, includeIntercept = TRUE) {
    if (is.null(varVec) && !includeIntercept)
      return(NULL)
    fm <- paste(attr(formula, "response"), "~")
    if (!is.null(varVec))
      fm <- paste(fm, paste(varVec, collapse = "+"))
    if (!includeIntercept)
      fm <- paste(fm, "-1 ")
    else fm <- paste(fm, "+1 ")
    modMat <- model.matrix(as.formula(fm), data)
  }
  X <- formDesignMat(varVec = attr(formula, "indSpVar"), includeIntercept = attr(formula,
                                                                                 "Intercept"))
  X <- if (!is.null(X))
    X[1:N, , drop = FALSE]
  Y <- formDesignMat(varVec = attr(formula, "csvChCoeff"),
                     includeIntercept = FALSE)
  Z <- formDesignMat(varVec = attr(formula, "csvGenCoeff"),
                     includeIntercept = FALSE)
  
  # Detect bad columns (these are linearly dependent on other columns)
  badColsList <- list(indSpVar = NULL, csvChCoeff = NULL, csvGenCoeff = NULL)
  getNullSpaceCols <- function(mat, tol = 1e-07) {
    if (is.null(mat)) {
      return(NULL)
    }
    if (dim(mat)[2] == 1) {
      return(NULL)
    }
    qrdecomp <- qr(mat, tol = tol)
    rank <- qrdecomp$rank
    if (rank == dim(mat)[2])  {
      return(NULL)
    }
    nullSpCols <- qrdecomp$pivot[(rank + 1):dim(mat)[2]]
    return(nullSpCols)
  }
  
  badColsList$indSpVar <- getNullSpaceCols(X, tol = linDepTol)
  for (i in 1:K) {
    init <- (i - 1) * N + 1
    fin <- i * N
    badColsList$csvChCoeff <- union(badColsList$csvChCoeff,
                                    getNullSpaceCols(Y[init:fin, , drop = FALSE], tol = linDepTol))
  }
  #do not drop Avl
  avlcol = which(colnames(Z) == "Avl")
  if (avlcol == 1) {
    badColsList$csvGenCoeff <- NULL
  } else {
    badColsList$csvGenCoeff <- getNullSpaceCols(Z[,-avlcol, drop = FALSE], tol = linDepTol)
  }
  
  # Get names of variables to be dropped from estimation
  badVarsList <- list()
  badVarsList$indSpVar <- colnames(X[, badColsList$indSpVar,
                                     drop = FALSE])
  badVarsList$csvChCoeff <- colnames(Y[, badColsList$csvChCoeff,
                                       drop = FALSE])
  badVarsList$csvGenCoeff <- colnames(Z[, badColsList$csvGenCoeff,
                                        drop = FALSE])
  badCoeffNames <- makeCoeffNames(badVarsList, choice.set)
  
  # Eliminate linearly dependent columns
  if (!is.null(X))
    X <- X[, setdiff(1:dim(X)[2], badColsList$indSpVar), drop = FALSE]
  if (!is.null(Y))
    Y <- Y[, setdiff(1:dim(Y)[2], badColsList$csvChCoeff),
           drop = FALSE]
  if (!is.null(Z))
    Z <- Z[, setdiff(1:dim(Z)[2], badColsList$csvGenCoeff),
           drop = FALSE]
  
  # Get names of variables
  varNamesList <- list()
  varNamesList$indSpVar <- colnames(X)
  varNamesList$csvChCoeff <- colnames(Y)
  varNamesList$csvGenCoeff <- colnames(Z)
  coeffNames <- makeCoeffNames(varNamesList, choice.set)
  if (!is.null(start))
    start[coeffNames] <- start
  baseChoiceName <- choice.set[1]
  if (!is.null(Z)) {
    for (ch_k in 2:K) {
      Z[((ch_k - 1) * N + 1):(ch_k * N), ] <- Z[((ch_k -
                                                    1) * N + 1):(ch_k * N), , drop = FALSE] - Z[1:N,
                                                                                                , drop = FALSE]
    }
  }
  Z <- Z[(N + 1):(K * N), , drop = FALSE]
  respVec <- respVec[(N + 1):(K * N)]
  t1 <- proc.time()[3]
  gc()
  prep.time <- t1 - startTime
  if (print.level > 1) {
    cat(paste0("Base alternative is: ", baseChoiceName))
    cat(paste0("\nPreprocessing data for estimation took ",
               round(prep.time, 3), " sec.\n"))
  }
  result <- newtonRaphson(respVec, X, Y, Z, K, maxiter, gtol,
                          ftol, ncores, print.level, coeffNames, weights = weights,
                          start = start)
  result$est.stats$prepTimeSecs <- prep.time
  colnames(result$hessMat) <- coeffNames
  rownames(result$hessMat) <- coeffNames
  names(result$grad) <- coeffNames
  od <- reordering(varNamesList, choice.set)
  coeffNames <- makeCoeffNames(varNamesList, choice.set)
  coefficients <- c(result$coeff, if (is.null(badCoeffNames)) NULL else rep(NA,
                                                                            length(badCoeffNames)))
  names(coefficients) <- c(coeffNames, badCoeffNames[reordering(badVarsList,
                                                                choice.set)])
  reordered_coeff <- c(result$coeff[od], if (is.null(badCoeffNames)) NULL else rep(NA,
                                                                                   length(badCoeffNames)))
  names(reordered_coeff) <- c(coeffNames[od], badCoeffNames[reordering(badVarsList,
                                                                       choice.set)])
  colnames(result$probability) <- choice.set
  if (maxiter > 0)
    colnames(result$residual) <- choice.set
  result$model.size$intercept <- interceptOn
  attributes(formula) <- NULL
  logLik <- structure(-result$loglikelihood, df = result$model.size$nparams,
                      class = "logLik")
  AIC <- 2 * (result$model.size$nparams + result$loglikelihood)
  index <- data.frame(chid = rep(1:result$model.size$N, result$model.size$K),
                      alt = data[[choiceVar]])
  attr(data, "index") <- index

  fit <- structure(list(coefficients = coefficients, logLik = logLik,
                        gradient = -result$grad, hessian = result$hessMat, est.stat = result$est.stats,
                        fitted.values = 1 - attr(result$residual, "outcome"),
                        probabilities = result$probability, residuals = result$residual,
                        df = result$model.size$nparams, AIC = AIC, choices = choice.set,
                        model.size = result$model.size, ordered.coeff = reordered_coeff,
                        model = data, freq = freq.choices, formula = Formula(formula(formula)),
                        call = initcall), class = "mnlogit")
  
  #to save memory, remove from model object all information unnecessary in prediction
  pf = parseFormula(fit$formula)
  fit <- structure(list(coefficients = fit$coefficients,
                        model.size = fit$model.size,
                        formula.response = attr(pf, "response"),
                        formula.indSpVar = attr(pf, "indSpVar"),
                        formula.Intercept = attr(pf, "Intercept"),
                        formula.csvChCoeff = attr(pf, "csvChCoeff"),
                        formula.csvGenCoeff = attr(pf, "csvGenCoeff"),
                        choiceSet.fit = unique(index(fit)$alt),
                        names.fit = names(fit$model),
                        badColsList.fit = badColsList), class = "mnlogit")

  if (print.level)
    cat(paste0("\nTotal time spent in mnlogit = ", round(proc.time()[3] -
                                                           startTime, 3), " seconds.\n"))
  #print(proc.time()[3] - startTime)
  return(fit)
}
parseFormula <- function (f)
{
  if (!is.Formula(f))
    f <- Formula(f)
  call <- formula(f)
  attr(call, "varNames") <- all.vars(f)
  numLHS <- length(f)[1]
  numRHS <- length(f)[2]
  if (numLHS != 1 && numRHS >= 1 && numRHS <= 3)
    stop("Invalid formula supplied.")
  lhsTerms <- terms(f, lhs = 1, rhs = 0)
  response <- all.vars(attr(lhsTerms, "variables"))
  if (length(response) != 1)
    stop("Invalid formula: response (LHS) can have only 1 term.")
  interceptON <- TRUE
  vars <- terms(f, lhs = 0, rhs = 1)
  x <- attr(vars, "term.labels")
  attr(call, "csvGenCoeff") <- if (length(x) > 0)
    x
  else NULL
  interceptON <- (interceptON && attr(vars, "intercept"))
  if (numRHS > 1) {
    vars <- terms(f, lhs = 0, rhs = 2)
    x <- attr(vars, "term.labels")
    attr(call, "indSpVar") <- if (length(x) > 0)
      x
    else NULL
    interceptON <- (interceptON && attr(vars, "intercept"))
  }
  if (numRHS > 2) {
    vars <- terms(f, lhs = 0, rhs = 3)
    x <- attr(vars, "term.labels")
    attr(call, "csvChCoeff") <- if (length(x) > 0)
      x
    else NULL
    interceptON <- (interceptON && attr(vars, "intercept"))
  }
  attr(call, "Intercept") <- interceptON
  attr(call, "response") <- response
  return(call)
}

predict.mnlogit <- function (object, newdata = NULL, probability = TRUE, returnData = FALSE,
                             choiceVar = NULL, ...)
{
  size <- object$model.size
  # get choice set for colnames
  choiceSet <- object$choiceSet.fit
  if (is.null(newdata)) {
    stop("Error! newdata must be specified")
  }
  else {
    # make sure newdata is ordered by choice
    if (is.null(choiceVar)) {
      if (!any(class(newdata) == "mlogit.data"))
        stop("NULL choiceVar requires newdata to be a mlogit.data object")
      if (nrow(newdata) != nrow(attr(newdata, "index")))
        stop("mlogit.data object newdata has incorrect index attribute")
      choiceVar <- "_Alt_Indx_"
      newdata[[choiceVar]] <- attr(newdata, "index")$alt
    }
    #newdata <- newdata[order(newdata[[choiceVar]]), ]
    newdata = as.data.table(newdata)
    setkeyv(newdata, choiceVar)
    newdata = as.data.frame(newdata)
    
    # Get name of response column
    #pf <- object$parsed.formula
    #resp.col <- attr(pf, "response")
    resp.col = object$formula.response
    
    # check that all columns from data are present (except response col)
    # this is important when you build Y below.
    newn <- names(newdata)
    oldn <- setdiff(object$names.fit, resp.col)
    if (!all(oldn %in% newn))
      stop("newdata must have same columns as training data. ")
    
    # different model size: N # newdata must have N*K rows
    if (nrow(newdata)%%size$K)
      stop("Mismatch between nrows in newdata and number of choices.")
  }
  data <- newdata
  size$N <- nrow(data)/size$K
  if (!(resp.col %in% names(data)))
    data[[resp.col]] <- rep(1, size$N)
  
  # Initialize utility matrix: dim(U) = N x K-1
  probMat <- matrix(rep(0, size$N * (size$K - 1)), nrow = size$N,
                    ncol = size$K - 1)
  formDesignMat <- function(varVec = NULL, includeIntercept = TRUE) {
    if (is.null(varVec) && !includeIntercept)
      return(NULL)
    fm <- paste(object$formula.response, "~")
    if (!is.null(varVec))
      fm <- paste(fm, paste(varVec, collapse = "+"))
    if (!includeIntercept)
      fm <- paste(fm, "-1 ")
    else fm <- paste(fm, "+1 ")
    modMat <- model.matrix(as.formula(fm), data)
  }
  # Grab the parsed formula from the fitted mnlogit object
  #formula <- object$parsed.formula
  X <- formDesignMat(varVec = object$formula.indSpVar, includeIntercept = object$formula.Intercept)
  X <- if (!is.null(X))
    X[1:size$N, , drop = FALSE]
  Y <- formDesignMat(varVec = object$formula.csvChCoeff,
                     includeIntercept = FALSE)
  Z <- formDesignMat(varVec = object$formula.csvGenCoeff,
                     includeIntercept = FALSE)
  
  badColsList = object$badColsList.fit
  if (!is.null(X))
    X <- X[, setdiff(1:dim(X)[2], badColsList$indSpVar), drop = FALSE]
  if (!is.null(Y))
    Y <- Y[, setdiff(1:dim(Y)[2], badColsList$csvChCoeff),
           drop = FALSE]
  if (!is.null(Z))
    Z <- Z[, setdiff(1:dim(Z)[2], badColsList$csvGenCoeff),
           drop = FALSE]
  
  # Do the subtraction: Z_ik - Zi0 (for Generic coefficients data)
  ### NOTE: Base choice (with respect to normalization) is fixed here
  ###       Base choice is the FIRST alternative
  if (!is.null(Z)) {
    for (ch_k in 2:size$K) {
      Z[((ch_k - 1) * size$N + 1):(ch_k * size$N), ] <- Z[((ch_k -
                                                              1) * size$N + 1):(ch_k * size$N), , drop = FALSE] -
        Z[1:size$N, , drop = FALSE]
    }
  }
  # Drop rows for base alternative
  Z <- Z[(size$N + 1):(size$K * size$N), , drop = FALSE]
  
  # Grab trained model coeffs from fitted mnlogit object
  coeffVec <- object$coeff
  # First compute the utility matrix (stored in probMat)
  if (size$p) {
    probMat <- probMat + X %*% matrix(coeffVec[1:((size$K -
                                                     1) * size$p)], nrow = size$p, ncol = (size$K - 1),
                                      byrow = FALSE)
  }
  if (size$f) {
    findYutil <- function(ch_k) {
      offset <- (size$K - 1) * size$p
      init <- (ch_k - 1) * size$N + 1
      fin <- ch_k * size$N
      Y[init:fin, , drop = FALSE] %*% coeffVec[((ch_k -
                                                   1) * size$f + 1 + offset):(ch_k * size$f + offset)]
    }
    vec <- as.vector(sapply(c(1:size$K), findYutil))
    # normalize w.r.t. to k0 here - see vignette on utility normalization
    vec <- vec - vec[1:size$N]
    probMat <- probMat + matrix(vec[(size$N + 1):(size$N *
                                                    size$K)], nrow = size$N, ncol = (size$K - 1), byrow = FALSE)
  }
  if (size$d) {
    probMat <- probMat + matrix(Z %*% coeffVec[(size$nparams -
                                                  size$d + 1):size$nparams], nrow = size$N, ncol = (size$K -
                                                                                                      1), byrow = FALSE)
  }
  # Convert utility to probabilities - use logit formula
  # probMat <- exp(probMat)
  # baseProbVec <- 1/(1 + rowSums(probMat))
  # probMat <- probMat * matrix(rep(baseProbVec, size$K -
  #                                   1), nrow = size$N, ncol = size$K - 1)
  probMat = cbind(rep(0, size$N), probMat)
  max.utilities = rowMaxs(probMat)
  probMat = probMat - max.utilities
  probMat = exp(probMat)
  row.sums.probmat = rowSums(probMat)
  probMat = probMat * matrix(rep(1/row.sums.probmat, size$K),
                             nrow = size$N, ncol = size$K)
  baseProbVec = probMat[,1]
  probMat = probMat[,2:size$K]
  
  probMat <- cbind(baseProbVec, probMat)
  if (nrow(probMat) == 1)
    probMat <- as.matrix(probMat)
  colnames(probMat) <- choiceSet
  if (probability) {
    if (returnData)
      attr(probMat, "data") <- newdata
    return(probMat)
  }
  else {
    stop("Error! probability must be TRUE")
  }
}
newtonRaphson <- function (response, X, Y, Z, K, maxiter, gtol, ftol, ncores,
                           print.level, coeff.names, weights = NULL, start = NULL)
{
  initTime <- proc.time()[3]
  # Determine problem parameters
  N <- length(response)/(K - 1) # num of observations
  p <- ifelse(is.null(X), 0, dim(X)[2]) # num of ind sp variables
  f <- ifelse(is.null(Y), 0, dim(Y)[2]) # num of ind sp variables
  d <- ifelse(is.null(Z), 0, dim(Z)[2]) # num of ind sp variables
  nparams <- (K - 1) * p + K * f + d # total number of coeffs
  size <- structure(list(N = N, K = K, p = p, f = f, d = d,
                         nparams = nparams), class = "model.size")
  
  # Find feature corresponding to availability indicators (assumed to be named "Avl")
  avl.ind = which(coeff.names == "Avl")
  # Define "large" constant for penalization of those alternatives not available in assortment
  M = 1e4
  
  # Initialize 'guess' for NR iteration
  #coeffVec <- if (!is.null(start)) start
  #else rep(0, size$nparams)
  if (!is.null(start)) {
    coeffVec = start
  } else {
    coeffVec = rep(0, size$nparams)
    coeffVec[avl.ind] = M #penalization of those alternatives not available in assortment
  }
  
  # Functions for correcting gradient and Hessian outputted by likelihood so that availability indicator constant isn't changed
  correctGrad <- function(gradient, avl.ind) {
    gradient = append(gradient, 0, after=(avl.ind-1))
    return(gradient)
  }
  correctHess <- function(Hessian, avl.ind) {
    #currently assumes Avl is last element of Z
    Hessian = rbind(Hessian, rep(0, dim(Hessian)[2]))
    Hessian = cbind(Hessian, rep(0, dim(Hessian)[1]))
    Hessian[avl.ind,avl.ind] = 1
    return(Hessian)
  }
  solvelinsys <- function(Hessian, gradient, eps=1e-3) {
    sol = Solve(Hessian, gradient)
    #sol = solve(Hessian, gradient, tol = 1e-24)
    
    #verify solution matches gradient
    # gradpred = Hessian %*% sol
    # if (mean(abs(gradient-as.vector(gradpred))) < eps) {
    #   return(sol)
    # } else {
    #   print(Hessian)
    #   print(gradient)
    #   print(as.vector(gradpred))
    #   print(sol)
    #   print(mean(abs(gradient-as.vector(gradpred))))
    #   #print(Solve(Hessian[1:4,1:4], gradient[1:4], tol = 1e-24))
    #   #print(Solve(Hessian[5:5,1:5], gradient[1:5], tol = 1e-24))
    #   stop("Error in solvelinsys(Hessian, gradient): Hessian %*% sol != gradient")
    # }
    return(sol)
  }
  # solvelinsys <- function(Hessian, gradient, tol=1e-10) {
  #   sol = Solve(Hessian, gradient, tol = 0)
  #   sol2 = solve(Hessian, gradient, tol = 0)
  #   sol3 = pinv(Hessian, tol = 0) %*% gradient
  #   #verify solution matches gradient
  #   gradpred = Hessian %*% sol
  #   gradpred2 = Hessian %*% sol2
  #   gradpred3 = Hessian %*% sol3
  #   
  #   print(mean(abs(gradient-as.vector(gradpred))))
  #   print(mean(abs(gradient-as.vector(gradpred2))))
  #   print(mean(abs(gradient-as.vector(gradpred3))))
  #   print("\n")
  #   if (mean(abs(gradient-as.vector(gradpred))) < tol) {
  #     return(sol)
  #   } else {
  #     #print(gradient)
  #     #print(as.vector(gradpred))
  #     #print(as.vector(solve(Hessian, gradient)))
  #     #print(as.vector(solve(Hessian, gradient, tol = 1e-24)))
  #     #print(Hessian)
  #     print("\n")
  #     stop("Error in solvelinsys(Hessian, gradient): Hessian %*% sol != gradient")
  #   }
  # }
  
  funcTime <- gradTime <- hessTime <- lineSearch <- solveTime <- 0.0
  
  # Compute log-likelihood, gradient, Hessian at initial guess
  fEval <- likelihood(response, X, Y, Z, size, coeffVec, ncores,
                      weights = weights)
  
  funcTime <- funcTime + attr(fEval, "funcTime")
  gradTime <- gradTime + attr(fEval, "gradTime")
  hessTime <- hessTime + attr(fEval, "hessTime")
  loglik <- fEval[[1]]
  gradient <- correctGrad(attr(fEval, "gradient"), avl.ind)
  hessian <- correctHess(attr(fEval, "hessian"), avl.ind)
  lineSearchIters <- 0
  failed.linesearch <- FALSE
  stop.code <- "null"
  if (maxiter < 1) {
    probMat <- attr(fEval, "probMat")
    #probMat <- cbind(1 - rowSums(probMat), probMat)
    if (is.vector(probMat)) {
      bProbVec <- 1 - probMat
    } else {
      bProbVec <- 1 - rowSums(probMat)
    }
    probMat <- cbind(bProbVec, probMat)
    residMat <- matrix(c(1:size$K), nrow = 1, ncol = size$K)
    return(list(coeff = coeffVec, loglikelihood = loglik,
                grad = gradient, hessMat = hessian, probability = probMat,
                residual = residMat, model.size = size, est.stats = NULL))
  }
  for (iternum in 1:maxiter) {
    oldLogLik <- loglik
    oldCoeffVec <- coeffVec
    t0 <- proc.time()[3]
    dir <- -1 * as.vector(solvelinsys(hessian, gradient))
    #dir <- -1 * as.vector(solve(hessian, gradient, tol = 1e-24))
    solveTime <- solveTime + proc.time()[3] - t0
    gradNorm <- as.numeric(sqrt(abs(crossprod(dir, gradient))))
    if (print.level) {
      cat("====================================================")
      cat(paste0("\nAt start of Newton-Raphson iter # ",
                 iternum))
      cat(paste0("\n  loglikelihood = ", round(loglik,
                                               8)))
      cat(paste0("\n  gradient norm = ", round(gradNorm,
                                               8)))
      cat("\n  Approx Hessian condition number = ")
      cat(paste0(round(1/rcond(hessian), 2), "\n"))
    }
    if (print.level > 1) {
      names(gradient) <- names(coeffVec) <- coeff.names
      coefTable <- rbind(coeffVec, gradient)
      rownames(coefTable) <- c("coef", "grad")
      print(coefTable, digits = 3)
      if (print.level > 2) {
        cat("\nPrinting the Hessian matrix.\n")
        colnames(hessian) <- rownames(hessian) <- coeff.names
        print(hessian, digits = 3)
      }
    }
    t1 <- proc.time()[3]
    alpha <- 1
    niter <- 0
    newloglik <- NULL
    while (1) {
      niter <- niter + 1
      coeffVec <- oldCoeffVec + alpha * dir
      newloglik <- likelihood(response, X, Y, Z, size,
                              coeffVec, ncores, hess = FALSE, weights = weights)
      funcTime <- funcTime + attr(newloglik, "funcTime")
      gradTime <- gradTime + attr(newloglik, "gradTime")
      hessTime <- hessTime + attr(newloglik, "hessTime")
      if (newloglik[[1]] < oldLogLik)
        break
      else alpha <- alpha/2
      if (max(abs(alpha * dir)) < 1e-15) {
        failed.linesearch <- TRUE
        break
      }
    }
    t2 <- proc.time()[3]
    lineSearchIters <- lineSearchIters + niter
    lineSearch <- lineSearch + t2 - t1
    
    # Compute new log-likelihood, gradient, Hessian
    fEval <- likelihood(response, X, Y, Z, size, coeffVec,
                        ncores, weights = weights, loglikObj = newloglik)
    funcTime <- funcTime + attr(fEval, "funcTime")
    gradTime <- gradTime + attr(fEval, "gradTime")
    hessTime <- hessTime + attr(fEval, "hessTime")
    loglik <- fEval[[1]]
    gradient <- correctGrad(attr(fEval, "gradient"), avl.ind)
    hessian <- correctHess(attr(fEval, "hessian"), avl.ind)
    
    # Success Criteria: function values converge
    loglikDiff <- abs(loglik - oldLogLik)
    if (loglikDiff < ftol) {
      stop.code <- paste0("Succesive loglik difference < ftol (",
                          ftol, ").")
      break
    }
    
    # Success Criteria: gradient reduces below gtol
    gradNorm <- as.numeric(sqrt(abs(crossprod(dir, gradient))))
    if (gradNorm < gtol) {
      stop.code <- paste0("Gradient norm < gtol (", gtol,
                          ").")
      break
    }
    # Success criterion not met, yet linesearch failed
    if (failed.linesearch) {
      stop.code <- paste0("Newton-Raphson Linesearch failed, can't do better")
      if (print.level) {
        print(paste("Failed linesearch: alpha, max(alpha * dir) = ",
                    alpha, max(abs(alpha * dir))))
      }
      break
    }
  } # end Newton-Raphson loop
  if (stop.code != "null" && iternum == maxiter)
    stop.code <- paste("Number of iterations:", iternum,
                       "== maxiters.")
  if (print.level) {
    cat("====================================================")
    cat(paste("\nTermination reason:", stop.code, "\n"))
  }
  stats <- structure(list(funcDiff = loglikDiff, gradNorm = gradNorm,
                          niters = iternum, LSniters = lineSearchIters, stopCond = stop.code,
                          totalMins = (proc.time()[3] - initTime)/60, hessMins = hessTime/60,
                          ncores = ncores), class = "est.stats")
  responseMat <- matrix(response, nrow = size$N, ncol = (size$K -
                                                           1))
  responseMat <- cbind(rep(1, size$N) - rowSums(responseMat),
                       responseMat)
  probMat <- attr(fEval, "probMat")
  #baseProbVec <- 1 - rowSums(probMat)
  if (is.vector(probMat)) {
    baseProbVec <- 1 - probMat
  } else {
    baseProbVec <- 1 - rowSums(probMat)
  }
  probMat <- cbind(baseProbVec, probMat)
  
  # Pch - prob of choice that was actually made
  Pch <- matrix(c(as.vector(ifelse(responseMat > 0, probMat,
                                   NA))), nrow = size$K, ncol = size$N, byrow = TRUE)
  Pch <- Pch[!is.na(Pch)]
  residMat <- responseMat - Pch # residual computation
  # Probability of NOT making the choice that was really made
  attr(residMat, "outcome") <- 1 - Pch
  result <- list(coeff = coeffVec, loglikelihood = loglik,
                 grad = gradient, hessMat = hessian, probability = probMat,
                 residual = residMat, model.size = size, est.stats = stats)
  return(result)
}
likelihood <- function (response, X, Y, Z, size, coeffVec, ncores, hess = TRUE,
                        weights = NULL, loglikObj = NULL)
{
  t0 <- t1 <- t2 <- proc.time()[3]
  
  if (is.null(loglikObj)) {
    # Initialize utility matrix: dim(U) = N x K-1
    probMat <- matrix(rep(0, size$N * (size$K - 1)), nrow = size$N,
                      ncol = size$K - 1)
    
    # First compute the utility matrix (stored in probMat)
    if (size$p) {
      probMat <- probMat + X %*% matrix(coeffVec[1:((size$K -
                                                       1) * size$p)], nrow = size$p, ncol = (size$K -
                                                                                               1), byrow = FALSE)
    }
    if (size$f) {
      findYutil <- function(ch_k) {
        offset <- (size$K - 1) * size$p
        init <- (ch_k - 1) * size$N + 1
        fin <- ch_k * size$N
        Y[init:fin, , drop = FALSE] %*% coeffVec[((ch_k -
                                                     1) * size$f + 1 + offset):(ch_k * size$f +
                                                                                  offset)]
      }
      vec <- as.vector(sapply(c(1:size$K), findYutil))
      vec <- vec - vec[1:size$N]
      probMat <- probMat + matrix(vec[(size$N + 1):(size$N *
                                                      size$K)], nrow = size$N, ncol = (size$K - 1),
                                  byrow = FALSE)
    }
    if (size$d) {
      probMat <- probMat + matrix(Z %*% coeffVec[(size$nparams -
                                                    size$d + 1):size$nparams], nrow = size$N, ncol = (size$K -
                                                                                                        1), byrow = FALSE)
    }
    
    # Compute partial log-likelihood
    loglik <- if (is.null(weights))
      drop(as.vector(probMat) %*% response)
    else drop(as.vector(probMat) %*% (weights * response))
    
    # Convert utility to probabilities - use logit formula
    # probMat <- exp(probMat)
    # baseProbVec <- 1/(1 + rowSums(probMat))
    # probMat <- probMat * matrix(rep(baseProbVec, size$K -
    #                                   1), nrow = size$N, ncol = size$K - 1)
    
    probMat = cbind(rep(0, size$N), probMat)
    max.utilities = rowMaxs(probMat)
    probMat = probMat - max.utilities
    probMat = exp(probMat)
    row.sums.probmat = rowSums(probMat)
    probMat = probMat * matrix(rep(1/row.sums.probmat, size$K),
                               nrow = size$N, ncol = size$K)
    baseProbVec = probMat[,1]
    probMat = probMat[,2:size$K]
    
    # Negative log-likelihood
    #loglik <- if (is.null(weights))
    #  -1 * (loglik + sum(log(baseProbVec)))
    #else -1 * (loglik + weights %*% log(baseProbVec))
    base.prob.denom = max.utilities + log(row.sums.probmat)
    if (is.null(weights)) {
      loglik <- -1 * (loglik - sum(base.prob.denom))
    } else {
      loglik <- -1 * (loglik - weights %*% base.prob.denom)
    }
  }
  else {
    loglik <- loglikObj
    attributes(loglik) <- NULL
    probMat <- attr(loglikObj, "probMat")
    #baseProbVec <- 1 - rowSums(probMat)
    if (is.vector(probMat)) {
      baseProbVec <- 1 - probMat
    } else {
      baseProbVec <- 1 - rowSums(probMat)
    }
  }
  t1 <- proc.time()[3] # Time after function evaluation
  
  #remove Avl from relevant features
  if (size$d == 1) {
    Z = NULL
  } else {
    avl.z.ind = which(colnames(Z) == "Avl")
    Z = Z[,-avl.z.ind, drop=FALSE]
  }
  size$d <- size$d - 1
  size$nparams <- size$nparams - 1
  
  # Gradient calculation
  gradient <- if (hess) {
    responseMat <- matrix(response, nrow = size$N, ncol = (size$K -
                                                             1))
    baseResp <- rep(1, size$N) - rowSums(responseMat)
    xgrad <- if (!is.null(X)) {
      if (is.null(weights))
        as.vector(crossprod(X, responseMat - probMat))
      else as.vector(crossprod(X, weights * (responseMat -
                                               probMat)))
    }
    else NULL
    ygrad <- if (!is.null(Y)) {
      yresp <- c((baseResp - baseProbVec), (response -
                                              as.vector(probMat)))
      findYgrad <- function(ch_k) {
        init <- (ch_k - 1) * size$N + 1
        fin <- ch_k * size$N
        if (is.null(weights))
          crossprod(Y[init:fin, , drop = FALSE], yresp[init:fin])
        else crossprod(Y[init:fin, , drop = FALSE], weights *
                         yresp[init:fin])
      }
      as.vector(sapply(c(1:size$K), findYgrad))
    }
    else NULL
    if (!is.null(Z)) {
      zgrad <- if (is.null(weights))
        as.vector(Z) * as.vector(responseMat - probMat)
      else as.vector(Z) * weights * as.vector(responseMat -
                                                probMat)
      zgrad <- colSums(matrix(zgrad, nrow = nrow(Z), ncol = size$d))
    }
    else zgrad <- NULL
    -1 * c(xgrad, ygrad, zgrad)
  }
  else attr(loglikObj, "gradient")
  t2 <- proc.time()[3]
  ans <- if (hess) {
    hessMat <- rep(0, size$nparams * size$nparams)
    .Call("computeHessianDotCall", as.integer(size$N), as.integer(size$K),
          as.integer(size$p), as.integer(size$f), as.integer(size$d),
          if (is.null(X)) NULL else as.double(t(X)), if (is.null(Y)) NULL else as.double(t(Y)),
          if (is.null(Z)) NULL else as.double(Z), if (is.null(weights)) NULL else as.double(weights),
          probMat, baseProbVec, as.integer(ncores), hessMat)
    hessMat
  }
  else NULL
  t3 <- proc.time()[3]
  attr(loglik, "probMat") <- probMat
  attr(loglik, "gradient") <- gradient
  attr(loglik, "hessian") <- if (!hess)
    attr(loglikObj, "hessian")
  else matrix(ans, nrow = size$nparams, ncol = size$nparams)
  attr(loglik, "funcTime") <- t1 - t0
  attr(loglik, "gradTime") <- t2 - t1
  attr(loglik, "hessTime") <- t3 - t2
  return(loglik)
}
assignInNamespace("likelihood",likelihood,ns="mnlogit")
assignInNamespace("newtonRaphson",newtonRaphson,ns="mnlogit")
assignInNamespace("predict.mnlogit",predict.mnlogit,ns="mnlogit")
assignInNamespace("mnlogit",mnlogit,ns="mnlogit")
assignInNamespace("makeCoeffNames",makeCoeffNames,ns="mnlogit")
assignInNamespace("reordering",reordering,ns="mnlogit")