%��ȡ����
load fisheriris;

%ֻ����%ȡǰ3ά
xdata = meas(51:end,3:4)

%���ݷ���
group = species(51:end);

%SVMѵ��
svmStruct = svmtrain(xdata,group,'ShowPlot',true);