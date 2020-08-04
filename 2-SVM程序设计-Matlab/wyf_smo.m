function [ alpha,bias ] = wyf_smo( training, groupIndex, boxconstraint, ...
                        tmp_kfun, smo_opts )
%wyf_smo smo�㷨��author:WangYifeng ,number:2019E8020261077
%   training�Ƕ�άѵ������
%   groupIndex�Ƕ�άѵ������training���ڵı�ǩ
%   boxconstraint��C��ֵ
%   tmp_kfun�Ǻ˺�����Ĭ�������Ժ˺���
%   smo_opts��smo�㷨������������
%           Display: 'off' --�Ƿ���ʾ
%           TolKKT: 1.0000e-03  --�����޸�
%           MaxIter: 15000  --��������
%           KKTViolationLevel: 0  --����KKT�����Ĳ���
%           KernelCacheLimit: 5000  --�˺����Ĵ�С���ƣ���
[m ,n]=size(training);
iter=0;
A=zeros(m,1);   %�����������Ƶ�alpha����
B=zeros(1,1);   %�����������Ƶ�bias

E=zeros(m,1);   %������������
Y=groupIndex;   %�����ı�ǩ
X=training;     %������ά����
Kernel_fun=tmp_kfun;    %�˺���
C=boxconstraint;    %C��alpha���Ͻ�
stop=0; %ͣ����־λ
%�����ʼ��������
for i=1:m
    E(i)=sum(A.*Y.*(Kernel_fun(X,X(i,:)))) + B-Y(i);
end

while iter<smo_opts.MaxIter
    iter=iter+1;    
    [A1_Index]=Select_a1(X,A,C,B,Y,E,smo_opts); %ѡ��a1������������A1_Index��
    [len_a1,~]=size(A1_Index);  %A1_Index���������� 
    
for i=1:len_a1
    index_a1=A1_Index(i);   %ȡ��A1_Index�ĵ�i������
    [A2_Index]=Select_a2(X,A,C,B,Y,E,index_a1,smo_opts);    %����a1ѡ��a2������������A2_Index��
    [len_a2,~]=size(A2_Index);  %A2_Index����������
    if len_a2==0 %ѡ��������ͣ��
        continue;
        len_a2
        stop=1;
    end
    
    for j=1:len_a2        
        index_a2=A2_Index(j);        
        [B,A,E]=Step(Kernel_fun,X,index_a2,index_a1,A,C,B,Y,E); %��������A,B,E
        [stop]=Check(E,Y,A);    %����Ƿ����ͣ������
        if stop==1
            alpha=A;
            bias=B;
            iter
            return;
        end            
    end
end
    
end
%���е�MaxIter��
alpha=A;
bias=B;
end

%����a2�����½磺L��H
function[L,H]=L_H(a1,a2,y1,y2,C)
        if y1~=y2
            kexi=a1-a2;            
            H=min(C,C-kexi);
            L=max(0,-kexi);           
        else
            kexi=a1+a2;
            H=min(C,kexi);
            L=max(0,kexi-C);
        end           
        
end

%����B,A,E
function[B,A,E]=Step(Kernel_fun,X,index_a2,index_a1,A,C,B,Y,E)        
        %�ȸ���a2
        E1=E(index_a1);
        E2=E(index_a2);
        a1=A(index_a1);
        a2=A(index_a2);
        mu=Kernel_fun(X(index_a2,:),X(index_a2,:))+Kernel_fun(X(index_a1,:),X(index_a1,:))-2*Kernel_fun(X(index_a1,:),X(index_a2,:));
        a2_new=a2+Y(index_a2)*(E1-E2)/mu;%%%%ɵ�ˣ�֮ǰд����abs
        %L-HԼ��
        y1=Y(index_a1);
        y2=Y(index_a2);
        [L, H]=L_H(a1,a2,y1,y2,C(1));       
        eta=1e-8;
        %a2�޸�����        
        if a2_new < eta
            a2_new = 0;
        elseif a2_new > C(1) - eta
            a2_new = C(1);
        end
        
        %a2����ֵ
        if a2_new<=H && a2_new>=L
            A(index_a2)=a2_new;
        elseif a2_new<L
            A(index_a2)=L;
        else
            A(index_a2)=H;
        end
    
        %a1����ֵ
        a1_new=a1+Y(index_a1)*Y(index_a2)*(a2-a2_new);
        
        %a1�޸����� 
        if a1_new<=C(1)-eta && a1_new>=eta
            A(index_a1)=a1_new;
        elseif a1_new<eta
            A(index_a1)=0;
        else
            A(index_a1)=C(1);
        end        
    
        %������ֵb
        K11=Kernel_fun(X(index_a1,:),X(index_a1,:));
        K12=Kernel_fun(X(index_a2,:),X(index_a1,:));
        K22=Kernel_fun(X(index_a2,:),X(index_a2,:));        
        if L==H            
            return
        end
        
        if index_a1==index_a2            
            return
        end
        %����B
        B_old=B;
        if a1_new>0 &&a1_new<C(1)
            b1_new=-E1-Y(index_a1).*K11.*(a1_new-a1)-Y(index_a2).*K12.*(a2_new-a2)+B;
            B=b1_new;      
        elseif a2_new>0 &&a2_new<C(1)
            b2_new=-E2-Y(index_a1).*K12.*(a1_new-a1)-Y(index_a2).*K22.*(a2_new-a2)+B;
            B=b2_new;        
        else         
            b1_new=-E1-Y(index_a1).*K11.*(a1_new-a1)-Y(index_a2).*K12.*(a2_new-a2)+B;
            b2_new=-E2-Y(index_a1).*K12.*(a1_new-a1)-Y(index_a2).*K22.*(a2_new-a2)+B;
            B=(b1_new+b2_new)/2;
        end
        
        %��ֹ�ָ��ߴ�ֱ��x�ᣬ����B��������������ܻ���������֡�
        if B>1e3
            B=1000;
        elseif B<-1e3
            B=-1000;
        end    
        
        %����a1,a2��Ӧ�����E1,E2
        E(index_a1)=sum(A.*Y.*(Kernel_fun(X,X(index_a1,:)))) + B-Y(index_a1);
        E(index_a2)=sum(A.*Y.*(Kernel_fun(X,X(index_a2,:)))) + B-Y(index_a2);
        
        %B���ˣ�����Eȫ��Ҫ�İ�����������������������
        [m ,~]=size(X);
        for i=1:m
            if (i~=index_a1 && i~=index_a2)
                E(i)=E(i)+B-B_old;
            end
        end       
        
end

%ѡ��a1
function[A1_Index]=Select_a1(X,A,C,B,Y,E,smo_opts)
A1_Index=[];
[m ,~]=size(X);
%���ж�0<ai<C�ĵ��Ƿ�Υ��KKT������Υ���ķ���A1_Index�С�
    for i=1:m
        if (A(i)>0) && (A(i)<C(i))
            if ((E(i)+Y(i))*Y(i)>1+smo_opts.TolKKT) || ((E(i)+Y(i))*Y(i)<1-smo_opts.TolKKT)
                A1_Index=[A1_Index;i];
            end
        end
    end
 %��0<ai<C�ĵ㲻Υ��KKT,����ai=0��ai=CΥ��KKT�����ĵ�
    [len,~]=size(A1_Index);
    if len==0
        for i=1:m
            if(A(i)>=0)
                if (E(i)+Y(i))*Y(i)<1-smo_opts.TolKKT
                    A1_Index=[A1_Index;i];
                end
            end
            if(A(i)<=C(1))
                if (E(i)+Y(i))*Y(i)>1+smo_opts.TolKKT
                    A1_Index=[A1_Index;i];
                end
            end
        end
        
    end
    
end

%����a1ѡ��a2
function[A2_Index]=Select_a2(X,A,C,B,Y,E,index_a1,smo_opts)
[m ,~]=size(X);
E1=E(index_a1);
A2_Index=[];
MAX=0;
delta=0.2;
a2_index=0;
%ppt�Ͳ��ϵķ���
% for i=1:m
%     E2=E(i);
%     if(abs(E2-E1)>delta)
%             a2_index=i;
%             A2_Index=[A2_Index;a2_index];
%     end
% end
% 
% if a2_index<1
%     for i=1:m
%         if A(i)>0 && A(i)<C(1)
%             a2_index=i;
%             A2_Index=[A2_Index;a2_index];
%         end
%     end
% end
% 
% if a2_index<1    
%     for i=1:m
%         a2_index=i;
%         A2_Index=[A2_Index;a2_index];
%     end
% end

%������ķ���
for i=1:m
    if A(i)>0 && A(i)<C(1)
        E2=E(i);
        if(abs(E2-E1)>MAX)
            a2_index=i;
            MAX=abs(E2-E1);
        end
    end 
    
end

if a2_index<1
        a2_index=randi(m);
end
A2_Index=[A2_Index;a2_index];

    
end

%�ж��Ƿ�ͣ��
function[stop]=Check(E,Y,A)
stop=0;
[m,~]=size(Y);
sum=0;
eta=0;
for i=1:m
    F=E(i)+Y(i);
    if (A(i)>0) && (A(i)<1)
        if(F*Y(i)>1-eta&&F*Y(i)<1+eta)
            sum=sum+1;
        end
    end
    if A(i)==0
        if F*Y(i)>=1
            sum=sum+1;
        end
    end
    if A(i)==1
        if F*Y(i)<=1
            sum=sum+1;
        end
    end
end
acc=sum/m;
if acc>0.95 %׼ȷ�ʴﵽ95%��ͣ��
    stop=1;
end
end

