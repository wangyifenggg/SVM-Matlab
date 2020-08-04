function [ alpha,bias ] = wyf_smo( training, groupIndex, boxconstraint, ...
                        tmp_kfun, smo_opts )
%wyf_smo smo算法，author:WangYifeng ,number:2019E8020261077
%   training是二维训练数据
%   groupIndex是二维训练数据training对于的标签
%   boxconstraint是C的值
%   tmp_kfun是核函数，默认是线性核函数
%   smo_opts是smo算法的其他参数：
%           Display: 'off' --是否显示
%           TolKKT: 1.0000e-03  --精度修复
%           MaxIter: 15000  --迭代次数
%           KKTViolationLevel: 0  --不符KKT条件的参数
%           KernelCacheLimit: 5000  --核函数的大小限制？？
[m ,n]=size(training);
iter=0;
A=zeros(m,1);   %根据样本估计的alpha矩阵
B=zeros(1,1);   %根据样本估计的bias

E=zeros(m,1);   %样本的误差矩阵
Y=groupIndex;   %样本的标签
X=training;     %样本二维数据
Kernel_fun=tmp_kfun;    %核函数
C=boxconstraint;    %C，alpha的上界
stop=0; %停机标志位
%计算初始的误差矩阵
for i=1:m
    E(i)=sum(A.*Y.*(Kernel_fun(X,X(i,:)))) + B-Y(i);
end

while iter<smo_opts.MaxIter
    iter=iter+1;    
    [A1_Index]=Select_a1(X,A,C,B,Y,E,smo_opts); %选择a1，索引保存在A1_Index中
    [len_a1,~]=size(A1_Index);  %A1_Index的索引个数 
    
for i=1:len_a1
    index_a1=A1_Index(i);   %取出A1_Index的第i个索引
    [A2_Index]=Select_a2(X,A,C,B,Y,E,index_a1,smo_opts);    %根据a1选择a2，索引保存在A2_Index中
    [len_a2,~]=size(A2_Index);  %A2_Index的索引个数
    if len_a2==0 %选不到可以停机
        continue;
        len_a2
        stop=1;
    end
    
    for j=1:len_a2        
        index_a2=A2_Index(j);        
        [B,A,E]=Step(Kernel_fun,X,index_a2,index_a1,A,C,B,Y,E); %迭代更新A,B,E
        [stop]=Check(E,Y,A);    %检查是否符合停机条件
        if stop==1
            alpha=A;
            bias=B;
            iter
            return;
        end            
    end
end
    
end
%运行到MaxIter次
alpha=A;
bias=B;
end

%计算a2的上下界：L和H
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

%更新B,A,E
function[B,A,E]=Step(Kernel_fun,X,index_a2,index_a1,A,C,B,Y,E)        
        %先更新a2
        E1=E(index_a1);
        E2=E(index_a2);
        a1=A(index_a1);
        a2=A(index_a2);
        mu=Kernel_fun(X(index_a2,:),X(index_a2,:))+Kernel_fun(X(index_a1,:),X(index_a1,:))-2*Kernel_fun(X(index_a1,:),X(index_a2,:));
        a2_new=a2+Y(index_a2)*(E1-E2)/mu;%%%%傻了，之前写多了abs
        %L-H约束
        y1=Y(index_a1);
        y2=Y(index_a2);
        [L, H]=L_H(a1,a2,y1,y2,C(1));       
        eta=1e-8;
        %a2修复精度        
        if a2_new < eta
            a2_new = 0;
        elseif a2_new > C(1) - eta
            a2_new = C(1);
        end
        
        %a2赋新值
        if a2_new<=H && a2_new>=L
            A(index_a2)=a2_new;
        elseif a2_new<L
            A(index_a2)=L;
        else
            A(index_a2)=H;
        end
    
        %a1赋新值
        a1_new=a1+Y(index_a1)*Y(index_a2)*(a2-a2_new);
        
        %a1修复精度 
        if a1_new<=C(1)-eta && a1_new>=eta
            A(index_a1)=a1_new;
        elseif a1_new<eta
            A(index_a1)=0;
        else
            A(index_a1)=C(1);
        end        
    
        %计算阈值b
        K11=Kernel_fun(X(index_a1,:),X(index_a1,:));
        K12=Kernel_fun(X(index_a2,:),X(index_a1,:));
        K22=Kernel_fun(X(index_a2,:),X(index_a2,:));        
        if L==H            
            return
        end
        
        if index_a1==index_a2            
            return
        end
        %更新B
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
        
        %防止分割线垂直于x轴，导致B无穷大的情况，尽管基本不会出现。
        if B>1e3
            B=1000;
        elseif B<-1e3
            B=-1000;
        end    
        
        %更新a1,a2对应的误差E1,E2
        E(index_a1)=sum(A.*Y.*(Kernel_fun(X,X(index_a1,:)))) + B-Y(index_a1);
        E(index_a2)=sum(A.*Y.*(Kernel_fun(X,X(index_a2,:)))) + B-Y(index_a2);
        
        %B变了，其他E全都要改啊！！！！！！！！！！！
        [m ,~]=size(X);
        for i=1:m
            if (i~=index_a1 && i~=index_a2)
                E(i)=E(i)+B-B_old;
            end
        end       
        
end

%选择a1
function[A1_Index]=Select_a1(X,A,C,B,Y,E,smo_opts)
A1_Index=[];
[m ,~]=size(X);
%先判断0<ai<C的点是否违反KKT条件，违反的放入A1_Index中。
    for i=1:m
        if (A(i)>0) && (A(i)<C(i))
            if ((E(i)+Y(i))*Y(i)>1+smo_opts.TolKKT) || ((E(i)+Y(i))*Y(i)<1-smo_opts.TolKKT)
                A1_Index=[A1_Index;i];
            end
        end
    end
 %若0<ai<C的点不违反KKT,再找ai=0和ai=C违反KKT条件的点
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

%根据a1选择a2
function[A2_Index]=Select_a2(X,A,C,B,Y,E,index_a1,smo_opts)
[m ,~]=size(X);
E1=E(index_a1);
A2_Index=[];
MAX=0;
delta=0.2;
a2_index=0;
%ppt和材料的方法
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

%例子里的方法
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

%判断是否停机
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
if acc>0.95 %准确率达到95%就停机
    stop=1;
end
end

