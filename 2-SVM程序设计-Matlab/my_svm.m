%Load the sample data, which includes Fisher's iris data of 5 measurements on a sample of 150 irises.
%加载matlab自带的一组分类数据
load fisheriris
%Create data, a two-column matrix containing sepal length and sepal width measurements for 150 irises.
%读取二维数据（2列数据）
data = [meas(:,1), meas(:,2)];

% From the species vector, create a new column vector, groups, to classify data into two groups: Setosa and non-Setosa.
%将数据
groups = ismember(species,'setosa');

%Randomly select training and test sets.
%随机选取训练与测试数据
[train, test] = crossvalind('holdOut',groups);
cp = classperf(groups);

%Train an SVM classifier using a linear kernel function and plot the grouped data.
%训练一个线性SVM分类器，并且画出支撑向量
svmStruct = my_svmtrain(data(train,:),groups(train),'showplot',true,'METHOD','SMO');
