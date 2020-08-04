function CPO = classperf(CP,varargin)
%CLASSPERF evaluates the performance of data classifiers
% 
%  CLASSPERF tracks performance measurements during the validation process
%  of classifiers. CLASSPERF creates and updates a CLASSPERFORMANCE object
%  which accumulates the results of the classifier. Classification
%  performance results can be accessed using the GET function or as fields in
%  structures. A CLASSPERFORMANCE object has the following properties:
%
%  Label:       User defined character string.
%  Description: User defined character string.
%  ClassLabels: Numeric vector or cell array of strings specifying a unique
%               set of class labels.
%  GroundTruth: Numeric vector or cell array of strings that specifies the
%               true class labels for each observation. The number of
%               elements is equal to NumberOfObservations.
%  NumberOfObservations: Positive integer specifying the number of
%               observations in the study.
%  ControlClasses: Indices to the ClassLabels vector or cell array,
%               indicating which classes to be considered as the control or
%               negative classes in a diagnostic test.
%  TargetClasses: Indices to the ClassLabels vector or cell array,
%               indicating which classes to be considered as the target or
%               positive classes in a diagnostic test.
%  ValidationCounter: Positive integer specifying the number of validations
%               performed.
%  SampleDistribution: Numeric vector indicating how many times each sample
%               was considered in the validation.
%  ErrorDistribution: Numeric vector indicating how many times each sample
%               was misclassified.
%  SampleDistributionByClass: Numeric vector indicating the frequency of
%               the true classes in the validation.
%  ErrorDistributionByClass: Numeric vector indicating the frequency of
%               errors for each class in the validation.
%  CountingMatrix: The classification confusion matrix. The order of rows
%               and columns is the same as grp2idx(GroundTruth). Columns
%               represent the true classes, and rows represent the
%               classifier prediction. The last row in CountingMatrix is
%               reserved to count inconclusive results. There are some
%               families of classifiers that can reserve the right to make
%               a hard class assignment; this can be based on metrics, such
%               as the posterior probabilities, or on how close a sample is
%               to the class boundaries.
%  CorrectRate: Correctly classified samples divided by the classified
%               samples. Inconclusive results are not counted.
%  ErrorRate:   Incorrectly classified samples divided by the
%               classified samples. Inconclusive results are not counted.
%  LastCorrectRate: Same as CorrectRate but only considering the samples
%               the last time the classifier performance object was updated.
%  LastErrorRate: Same as LastCorrectRate but only considering the samples
%               the last time the classifier performance object was updated.
%  InconclusiveRate: Nonclassified samples divided by the total number of
%               samples.
%  ClassifiedRate: Classified samples divided by the total number of
%               samples.
%  Sensitivity: Correctly classified positive samples divided by true
%               positive samples. Inconclusive results that are true
%               positives are counted as errors for computing Sensitivity
%               (following a conservative approach). This is the same as
%               being incorrectly classified as negatives.
%  Specificity: Correctly classified negative samples divided by the true
%               negative samples. Inconclusive results that are true
%               negatives are counted as errors for computing Specificity
%               (following a conservative approach). This is the same as
%               being incorrectly classified as positives.
%  PositivePredictiveValue: Correctly classified positive samples divided
%               by the positive classified samples. Inconclusive results
%               are classified as negatives when computing
%               PositivePredictiveValue.
%  NegativePredictiveValue: Correctly classified negative samples divided
%               by the negative classified samples. Inconclusive results
%               are classified as positives when computing
%               NegativePredictiveValue.
%  PositiveLikelihood:  Sensitivity / (1 - Specificity)
%  NegativeLikelihood:  (1 - Sensitivity) / Specificity
%  Prevalence:  True positive samples divided by the total number of
%               samples.
%  DiagnosticTable: A 2-by-2 numeric array with diagnostic counts. The
%               first row indicates the number of samples that were
%               classified as positive, with the number of true positives
%               in the first column, and the number of false positives in
%               the second column. The second row indicates the number of
%               samples that were classified as negative, with the number
%               of false negatives in the first column, and the number of
%               true negatives in the second column.
%               Correct classifications appear in the diagonal elements,
%               and errors appear in the off-diagonal elements.
%               Inconclusive results are considered errors and counted in
%               the off-diagonal elements.
%
%  CP = CLASSPERF(GROUNDTRUTH) creates and initializes an empty classifier
%  performance object. CP is the handle to the object. GROUNDTRUTH is a
%  vector with the true class labels for every observation. GROUNDTRUTH can
%  be either a numeric vector or a cell array of strings. When used in a
%  cross-validation design experiment, GROUNDTRUTH should have the same
%  size as the total number of observations.
%
%  CLASSPERF(CP,CLASSOUT) updates the CP object with the classifier output
%  CLASSOUT. CLASSOUT is the same size and type as GROUNDTRUTH. When
%  CLASSOUT is numeric and GROUNDTRUTH is a cell array of strings, GRP2IDX
%  is used to create the index vector that links CLASSOUT to the class
%  labels. When CLASSOUT is a cell array of strings, an empty string, '',
%  represents an inconclusive result of the classifier. For numeric arrays,
%  NaN represents an inconclusive result. 
%
%  CLASSPERF(CP,CLASSOUT,TESTIDX) updates the CP object with the classifier
%  output CLASSOUT. CLASSOUT is smaller than GROUNDTRUTH. TESTIDX is an
%  index vector or a logical index vector of the same size as GROUNDTRUTH
%  that indicates which observations were used in the current validation.
%
%  CP = CLASSPERF(GROUNDTRUTH,CLASSOUT,...) creates and updates the CP
%  object with the first validation. This form is useful when you want to
%  know the performance of a single validation.
%
%  CP = CLASSPERF(...,'POSITIVE',P,'NEGATIVE',N) sets the 'POSITIVE' and
%  'NEGATIVE' labels to identify the target disorder and the control
%  classes. These labels are used to compute clinical diagnostic test
%  performance. P and N are disjoint subsets of UNIQUE(GROUNDTRUTH) and
%  they are the same type of input as GROUNDTRUTH. When omitted, P defaults
%  to the first class returned by GRP2IDX(GROUNDTRUTH) and N defaults to
%  all the others. In clinical tests, inconclusive values ('' or NaN) are
%  counted as false negatives for the computation of the specificity and as
%  false positives for the computation of the sensitivity. I.e.,
%  inconclusive results can decrease the diagnostic value of the test.
%  Tested observations whose true class is not within the union of P and N
%  are ignored. Tested observations whose true class is within the union of
%  P and N but were classified outside the union are counted as
%  inconclusive.
%
%  Examples:
%
%     % Classify the fisheriris data with a K-Nearest Neighbor classifier.
%     load fisheriris
%     c = knnclassify(meas,meas,species,4,'euclidean','Consensus');
%     cp = classperf(species,c)
%
%     % Create a 10-fold cross-validation on the fisheriris data using
%     % linear discriminant analysis and the third column as the only
%     % feature for classification.
%     load fisheriris
%     indices = crossvalind('Kfold',species,10);
%     cp = classperf(species); % initializes the CP object
%     for i = 1:10
%         test = (indices == i); train = ~test;
%         class = classify(meas(test,3),meas(train,3),species(train));
%         % updates the CP object with the current classification results
%         classperf(cp,class,test);
%     end
%     cp.CorrectRate % queries for the correct classification rate
%
%
%  See also CANCERDETECTDEMO, CROSSVALIND, CLASSIFY, GRP2IDX, KNNCLASSIFY,
%  SVMCLASSIFY.

% Copyright 2003-2010 The MathWorks, Inc.
% $Revision: 1.1.10.8 $  $Date: 2010/05/10 16:57:49 $

% References:
% [1] Nicoll D, McPhee SJ, Pigone M, Detmer WM, Chou TM. Pocket Guide to
%     Diagnostic Tests, Third Edition  University of California, San
%     Francisco   
%
% [2] Theodoridis, S. and Koutroumbas, K.  (1999) Pattern Recognition,
%     Academic Press, pp. 341-342.

if nargin == 0
    if nargout 
        error('Bioinfo:classperf:TooManyOutputArguments',...
        'When there are not input arguments, CLASSPERF cannot have output arguments.')
    end
    displayProperties
    return
end

% set some defaults
gpsWereGiven = false;

% if the first input is the GROUNDTRUTH we need to create and initialize
% the new CP object
if ~isequal(class(CP),'biolearning.classperformance')
    if isa(CP,'ordinal')||isa(CP,'nominal')
        CP = cellstr(CP(:));
    end
    % validate ground truth labels
    gT = CP(:);
    if iscell(gT)
        if any(~cellfun('isclass',gT,'char'))
            error('Bioinfo:classperf:InvalidCellForGT',...
                'Ground truth cell must be all strings.')
        end
        if ismember('',gT)
            error('Bioinfo:classperf:EmptyCellForGT',...
                'Empty strings are not allowed in the ground truth vector.')
        end
        [igT,validGroups] = grp2idx(gT);
    elseif isnumeric(gT) || islogical(gT)
        if islogical(gT)
            gT = double(gT);
        end
        if ~isreal(gT) || any(isnan(gT)) || any(rem(gT,1)) 
            error('Bioinfo:classperf:InvalidNumericForGT',...
                'Ground truth array is not valid.')
        end
        [validGroups,~,igT] = unique(gT); 
    else
        error('Bioinfo:classperf:InvalidTypeForGT',...
            'Ground truth must be a cell of strings or a numeric array.')
    end
    if max(igT)<2
         error('Bioinfo:classperf:InvalidGroupsForGT',...
            'Ground truth must have at least two classes.')
    end
    % Creates and initializes the CP object
    CP = biolearning.classperformance(validGroups,igT);
end

if nargin == 1 
    CPO=CP; 
    return; 
end
nvarargin =  numel(varargin);

% check if the second input is CLASSOUT and validate it
if ~ischar(varargin{1})
    gps = varargin{1}(:);
    if isa(gps,'ordinal')||isa(gps,'nominal')
        gps = cellstr(gps(:));
    end
    nvarargin = nvarargin-1;
    varargin(1) = [];
    if CP.IsClassLabelTypeNumeric
        if ~isnumeric(gps) || ~isreal(gps) || any(gps(~isnan(gps))<0) || any(rem(gps(~isnan(gps)),1))
            error('Bioinfo:classperf:InvalidNumericForGRP',...
                ['When the class labels of the CP object are numeric, the output\n'...
                'of the classifier must be all non-negative integers or NaN''s.'])
        end
        [~,gps] = ismember(gps,CP.ClassLabels);
    elseif iscell(gps)
        if any(~cellfun('isclass',gps(:),'char'))
            error('Bioinfo:classperf:InvalidCellForGRP',...
                  ['When the classifier output is a cell array of strings,\n',...
                   'all elements must have strings.'])
        end
        [~,gps] = ismember(gps,CP.ClassLabels);
    elseif isnumeric(gps)
        if ~isreal(gps) || any(gps(~isnan(gps))<0) || any(rem(gps(~isnan(gps)),1))
            error('Bioinfo:classperf:InvalidIndicesForGRP',...
                ['When class labels of the CP object is a cell array of strings and\n',...
                 'the classifier output is a numeric array, it must contain valid\n',...
                 'indices of the class labels or NaNs for inconclusive results.'])
        end
    else
        error('Bioinfo:classperf:InvalidTypeForGRP',...
              ['CLASSOUT should be the same type as the ground truth vector\n'...
               'or a vector index to the class labels.'])
    end
    gpsWereGiven = true;
end

% check if the third input is TESTIDX 
if nvarargin && (islogical(varargin{1}) || isnumeric(varargin{1}))
    idx = varargin{1}(:);
    nvarargin = nvarargin-1;
    varargin(1) = [];
elseif gpsWereGiven
    if numel(gps)~=CP.NumberOfObservations
        error('Bioinfo:classperf:IncorrectSizeForClassout',...
            ['The classifier output CLASSOUT does not have the same size\n',...
            'as the ground truth and there was not any TESTIDX provided.']);        
    end
    idx = 1:CP.NumberOfObservations;
end

% the rest should can only be optional arguments (i.e. set
% negative/positive class labels)
if nvarargin
    positiveLabels = CP.TargetClasses;
    negativeLabels = CP.ControlClasses;
    if rem(nvarargin,2)
        error('Bioinfo:classperf:IncorrectNumberOfArguments',...
            'Incorrect number of arguments to %s.',mfilename);
    end
    okargs = {'positive','negative'};
    for j=1:2:nvarargin
        pname = varargin{j};
        pval = varargin{j+1};
        k = find(strncmpi(pname, okargs,length(pname)));
        if isempty(k)
            error('Bioinfo:classperf:UnknownParameterName',...
                'Unknown parameter name: %s.',pname);
        elseif length(k)>1
            error('Bioinfo:classperf:AmbiguousParameterName',...
                'Ambiguous parameter name: %s.',pname);
        else
            switch(k)
                case 1 % positiveLabels
                    if isa(pval,'ordinal')||isa(pval,'nominal')
                        pval = cellstr(pval(:));
                    end
                    if CP.IsClassLabelTypeNumeric
                        [tf,loc] = ismember(pval,CP.ClassLabels);
                        if any(~tf) || isempty(tf)
                            error('Bioinfo:classperf:InvalidPositiveLabels',...
                                'At least one of the positive labels is not within the valid class labels.')
                        end
                        positiveLabels = loc;
                    else
                        if ischar(pval)
                            pval = {pval};
                        end
                        if ~iscell(pval) || ~all(cellfun('isclass',pval,'char'))
                            error('Bioinfo:classperf:InvalidPositiveLabels',...
                                'Invalid type for the positive labels.')
                        end
                        [tf,loc] = ismember(pval,CP.ClassLabels);
                        if any(~tf) || isempty(tf)
                            error('Bioinfo:classperf:InvalidPositiveLabels',...
                                'At least one of the positive labels is not within the valid class labels.')
                        end
                        positiveLabels = loc;
                    end
                case 2 % negativeLabels
                    if isa(pval,'ordinal')||isa(pval,'nominal')
                        pval = cellstr(pval(:));
                    end
                    if CP.IsClassLabelTypeNumeric
                        [tf,loc] = ismember(pval,CP.ClassLabels);
                        if any(~tf) || isempty(tf)
                            error('Bioinfo:classperf:InvalidNegativeLabels',...
                                'At least one of the negative labels is not within the valid class labels.')
                        end
                        negativeLabels = loc;
                    else
                        if ischar(pval)
                            pval = {pval};
                        end
                        if ~iscell(pval) || ~all(cellfun('isclass',pval,'char'))
                            error('Bioinfo:classperf:InvalidNegativeLabels',...
                                'Invalid type for negative labels.')
                        end
                        [tf,loc] = ismember(pval,CP.ClassLabels);
                        if any(~tf) || isempty(tf)
                            error('Bioinfo:classperf:InvalidNegativeLabels',...
                                'At least one of the negative labels is not within the valid class labels.')
                        end
                        negativeLabels = loc;
                    end
            end
        end
    end
    % check that diagnostic test labels do not collide
    if ~isempty(intersect(positiveLabels,negativeLabels))
        error('Bioinfo:classperf:InvalidPositiveNegativeLabels',...
            'Positive and negative class labels must be disjoint sets.')
    else
        CP.TargetClasses  = positiveLabels;
        CP.ControlClasses = negativeLabels;
    end
end

% I am done creating the object with new pos and neg, now I can leave
if ~gpsWereGiven
    CPO = CP;
    return
end

% validate idx
if islogical(idx)
    if numel(idx)~=CP.NumberOfObservations
        error('Bioinfo:classperf:InvalidLogicalIndex',...
              'Size of the logical index vector must be [NumberOfObservations x 1].')
    end
    if sum(idx)~=numel(gps)
        error('Bioinfo:classperf:InvalidLogicalIndex',...
              'There must be as many TRUE indices in TESTIDX as classifier outputs.')
    end
    idx = find(idx);
else % it is a vector of indices
    if numel(unique(idx))~=numel(idx) || max(idx)>CP.NumberOfObservations || min(idx)<0 || any(rem(idx,1)>0)
        error('Bioinfo:classperf:InvalidNumericIndex',...
              'Index vector has invalid values.')
    end
    if numel(idx)~=numel(gps)
        error('Bioinfo:classperf:InvalidNumericIndex',...
              'There must be as many indices in TESTIDX as classifier outputs.')
    end
end

% CP and all input arguments should be valid now, do the update
CP.updateValidation(gps,idx);
    
CPO = CP; 

function displayProperties
blpk = findpackage('biolearning');
cls = findclass(blpk,'classperformance');
p = get(cls,'Properties');
names = get(p,'Name');
disp('CLASSPERFORMANCE object public properties:')
disp(names(strcmp(get(p,'Visible'),'on')))


