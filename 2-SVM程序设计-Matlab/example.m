%读取数据
load fisheriris;

%只保留%取前3维
xdata = meas(51:end,3:4)

%数据分组
group = species(51:end);

%SVM训练
svmStruct = svmtrain(xdata,group,'ShowPlot',true);