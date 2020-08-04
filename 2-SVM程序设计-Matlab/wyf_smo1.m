function [ alpha,bias ] = wyf_smo1( training, groupIndex, boxconstraint, ...
                        tmp_kfun, smo_opts )
%wyf_smo smo算法
%   training是二维训练数据
%   groupIndex是二维训练数据training对于的标签
%   boxconstraint是C的值
%   tmp_kfun是核函数，默认是线性核函数
%   smo_opts是smo算法的其他参数：
%           Display: 'off' --是否显示
%           TolKKT: 1.0000e-03  --KKT条件精度
%           MaxIter: 15000  --迭代次数
%           KKTViolationLevel: 0  --不符KKT条件的参数
%           KernelCacheLimit: 5000  --核函数的大小限制
[m ,n]=size(training);
iter=0;
A=zeros(m,1);%m列1行
B=zeros(1,1);

E=zeros(m,1);
Y=groupIndex;
X=training;
Kernel_fun=tmp_kfun;
C=boxconstraint;
for i=1:m
    E(i)=sum(A.*Y.*(Kernel_fun(X,X(i,:)))) + B-Y(i);
end
Pre_A1_Index=[];
stop=0;
while iter<smo_opts.MaxIter-14900||(stop>0)
    iter=iter+1;
    
    [A1_Index,Pre_A1_Index,stop]=Select_a1(X,A,C,B,Y,E,Kernel_fun,smo_opts,Pre_A1_Index);
    [len_a1,~]=size(A1_Index);
    
    if stop==1
        break;
    end

for i=1:len_a1
    index_a1=A1_Index(i);
    [A2_Index]=Select_a22(X,A,C,B,Y,E,Kernel_fun,index_a1,smo_opts);
    [len_a2,~]=size(A2_Index);
    if len_a2==0
        continue;
        len_a2
        stop=1;
    end
    
    for j=1:len_a2        
        index_a2=A2_Index(j);        
        [B,A,E]=Step(Kernel_fun,X,index_a2,index_a1,A,C,B,Y,E);
    end
end
    
end

alpha=A;
bias=B;
end

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

function[B,A,E]=Step(Kernel_fun,X,index_a2,index_a1,A,C,B,Y,E)
        %更新a2
        E1=E(index_a1);
        E2=E(index_a2);
        a1=A(index_a1);
        a2=A(index_a2);
        mu=Kernel_fun(X(index_a2,:),X(index_a2,:))+Kernel_fun(X(index_a1,:),X(index_a1,:))-2*Kernel_fun(X(index_a1,:),X(index_a2,:));
        a2_new=a2+Y(index_a2)*(E1-E2)/mu;%%%%傻了，写多了abs
        %L-H约束
        y1=Y(index_a1);
        y2=Y(index_a2);
        [L, H]=L_H(a1,a2,y1,y2,C(1));%%%%%%应该使用a2还是a2_new？？？       
        eta=1e-8;
        %补充修复精度        
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
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%约束一样？？？
        if a1_new<=C(1)-eta && a1_new>=eta
            A(index_a1)=a1_new;
        elseif a1_new<eta
            A(index_a1)=0;
        else
            A(index_a1)=C(1);
        end
        %A(index_a1)=a1_new;
    
        %计算阈值b
        K11=Kernel_fun(X(index_a1,:),X(index_a1,:));
        K12=Kernel_fun(X(index_a2,:),X(index_a1,:));
        K22=Kernel_fun(X(index_a2,:),X(index_a2,:));
        B_old=B;
        if a1_new>0 &&a1_new<C(1)
            b1_new=-E1-Y(index_a1).*K11.*(a1_new-a1)-Y(index_a2).*K12.*(a2_new-a2)+B;
            B=b1_new;
        
        
        elseif a2_new>0 &&a2_new<C(1)
            b2_new=-E2-Y(index_a1).*K12.*(a1_new-a1)-Y(index_a2).*K22.*(a2_new-a2)+B;
            B=b2_new;
        
        else          %??????????????????
            b1_new=-E1-Y(index_a1).*K11.*(a1_new-a1)-Y(index_a2).*K12.*(a2_new-a2)+B;
            b2_new=-E2-Y(index_a1).*K12.*(a1_new-a1)-Y(index_a2).*K22.*(a2_new-a2)+B;
            B=(b1_new+b2_new)/2;
        end
    
        
        
        E(index_a1)=sum(A.*Y.*(Kernel_fun(X,X(index_a1,:)))) + B-Y(index_a1);
        E(index_a2)=sum(A.*Y.*(Kernel_fun(X,X(index_a2,:)))) + B-Y(index_a2);
        %B变了，E全都要改啊！！！！！！！！！！！
        [m ,~]=size(X);
        for i=1:m
            if (i~=index_a1 && i~=index_a2)
                E(i)=E(i)+B-B_old;
            end
        end
end

function[A1_Index,Pre_A1_Index,stop]=Select_a1(X,A,C,B,Y,E,Kernel_fun,smo_opts,Pre_A1_Index)
A1_Index=[];
stop=0;
[m ,~]=size(X);
    for i=1:m
        if (A(i)>smo_opts.TolKKT) && (A(i)<C(i))
            if (E(i)+Y(i))*Y(i)~=1
                A1_Index=[A1_Index;i];
            end
        end
    end
 
    [len,~]=size(A1_Index);
    if len==0
        for i=1:m
            if(A(i)<=smo_opts.TolKKT)
                if (E(i)+Y(i))*Y(i)<1
                    A1_Index=[A1_Index;i];
                end
            end
            if(A(i)>=C(1))
                if (E(i)+Y(i))*Y(i)>1
                    A1_Index=[A1_Index;i];
                end
            end
        end
        
    end
    [len,~]=size(A1_Index);
    if len==0
        stop=1;
    end
    
    
    %Pre_A1_Index=A1_Index;
    

end

function[A2_Index]=Select_a2(X,A,C,B,Y,E,Kernel_fun,index_a1,smo_opts)
[m ,~]=size(X);
[~,Index_E]=sort(E);
E1=E(index_a1);
A2_Index=[];
change_a1=0;
if(E1>0)
    
            for i=1:m
                index_a2=Index_E(i); 
                if (A(index_a2)>smo_opts.TolKKT) && (A(index_a2)<C(index_a2))
                
                E2=E(index_a2);
                if abs(E1-E2)>smo_opts.TolKKT
                    A2_Index=[A2_Index;index_a2];
                end
                %[B,A,C,E]=Step(Kernel_fun,X,index_a2,index_a1,E1,E2,A,C,B,Y,a1,a2,E);
                end
            end
    
 else
            for i=m:1
                index_a2=Index_E(i); 
                if (A(index_a2)>smo_opts.TolKKT) && (A(index_a2)<C(index_a2))
                E2=E(index_a2);
                if abs(E1-E2)>smo_opts.TolKKT
                    A2_Index=[A2_Index;index_a2];
                end
                %[B,A,C,E]=Step(Kernel_fun,X,index_a2,index_a1,E1,E2,A,C,B,Y,a1,a2,E);
                end
            end
end
[len,~]=size(A2_Index);


if len==0%把所有点放进来迭代
    if(E1>0)    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for i=1:m
                index_a2=Index_E(i);                     
                E2=E(index_a2);
                if abs(E1-E2)>smo_opts.TolKKT
                    A2_Index=[A2_Index;index_a2];
                end
                %[B,A,C,E]=Step(Kernel_fun,X,index_a2,index_a1,E1,E2,A,C,B,Y,a1,a2,E);
                
            end
    
    else
            for i=m:-1:1%%%%%%%%
                index_a2=Index_E(i);                 
                E2=E(index_a2);
                if abs(E1-E2)>smo_opts.TolKKT
                    A2_Index=[A2_Index;index_a2];
                end
                %[B,A,C,E]=Step(Kernel_fun,X,index_a2,index_a1,E1,E2,A,C,B,Y,a1,a2,E);
                
            end
    end
end


[len,~]=size(A2_Index);
if len==0
    change_a1=1;
end



end

function[A2_Index]=Select_a22(X,A,C,B,Y,E,Kernel_fun,index_a1,smo_opts)
[m ,~]=size(X);
E1=E(index_a1);
A2_Index=[];
MAX=0;
a2_index=0;
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


     

    

