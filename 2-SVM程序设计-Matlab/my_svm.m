%Load the sample data, which includes Fisher's iris data of 5 measurements on a sample of 150 irises.
%����matlab�Դ���һ���������
load fisheriris
%Create data, a two-column matrix containing sepal length and sepal width measurements for 150 irises.
%��ȡ��ά���ݣ�2�����ݣ�
data = [meas(:,1), meas(:,2)];

% From the species vector, create a new column vector, groups, to classify data into two groups: Setosa and non-Setosa.
%������
groups = ismember(species,'setosa');

%Randomly select training and test sets.
%���ѡȡѵ�����������
[train, test] = crossvalind('holdOut',groups);
cp = classperf(groups);

%Train an SVM classifier using a linear kernel function and plot the grouped data.
%ѵ��һ������SVM�����������һ���֧������
svmStruct = my_svmtrain(data(train,:),groups(train),'showplot',true,'METHOD','SMO');
