function [alpha, bias] = my_smo(training, groupIndex, boxconstraint, ...
                        tmp_kfun, smo_opts)
global m;
global X;
global Y;
global C;
global B;
global C_max;
global kernel;
global opts;
[m,~] = size(training);
X = training;
Y = groupIndex;
kernel = tmp_kfun;
opts = smo_opts;
C_max = boxconstraint;
C = zeros(m, 1);
B = zeros(1, 1);
numChanged = 0;%numChanged=0且examineAll = 0时停机
examineAll = 1;%是否遍历全部,0表示没遍历全部
step = 0;
while (numChanged > 0 || examineAll) && step < opts.MaxIter
    numChanged = 0; 
    if examineAll
        for jj = 1:m
            numChanged = numChanged + examineExample(jj)
        end
    else
        for jj = 1:m
            if C(jj) > 0 && C(jj) < C_max(jj)
                numChanged = numChanged + examineExample(jj)
            end
        end
    end
    if examineAll == 1
        examineAll = 0;
    elseif numChanged == 0
        examineAll = 1;
    end
    if (numChanged > 0 || examineAll)==0%全检查了examineAll=1，赋值为0，然后numChanged等于0
        numChanged;
        examineAll;
    end
    step = step + 1
end
alpha = C;
bias = -B;
end

function[out] = takeStep(ii,jj)
global X;
global Y;
global C;
global B;
global C_max;
global kernel;
global opts;
if ii == jj
    out = 0;
else
    E1 = sum(C.*Y.*(kernel(X,X(ii,:)))) - B - Y(ii);
    E2 = sum(C.*Y.*(kernel(X,X(jj,:)))) - B - Y(jj);
    if Y(ii) == Y(jj)
        L = max(0,(C(ii) + C(jj) - C_max(ii)));
        H = min(C_max(ii),(C(ii) + C(jj)));
    else
        L = max(0,(C(jj) - C(ii)));
        H = min(C_max(ii),(C_max(ii) + C(jj) - C(ii)));
    end
    if L == H
        out = 0;
    else
        s = Y(ii) * Y(jj);
        kii = kernel(X(ii,:),X(ii,:));
        kij = kernel(X(ii,:),X(jj,:));
        kjj = kernel(X(jj,:),X(jj,:));
        eta = 2 * kij - kii - kjj;
        if eta < 0
            Temp =  C(jj) - Y(jj) * (E1 - E2) / eta;
            C2 = max(min(Temp, H), L);
        else                      %%%%%%%%%%%%%%%%%%%%%%%%？？？？？？？？？？？？？？？？？？？？
%             Lc1 = C(ii) + s * (C(jj) - L);
%             Lobj = Lc1 + L + 0.5 * s * kernel(X(ii,:),X(jj,:)) * Lc1 * L;
%             Hc1 = C(ii) + s * (C(jj) - H);
%             Hobj = Hc1 + H + 0.5 * s * kernel(X(ii,:),X(jj,:)) * Hc1 * H;
%             if Lobj > Hobj + opts.KKTViolationLevel
%                 C2 = L;
%             elseif Lobj < Hobj - opts.KKTViolationLevel
%                 C2 = H;
%             else
%                 C2 = C(jj);
%             end
            Temp =  C(jj) - Y(jj) * (E1 - E2) / eta;
            C2 = max(min(Temp, H), L);
        end
        if C2 < 1e-8
            C2 = 0;
        elseif C2 > C_max(jj) - 1e-8
            C2 = C_max(jj);
        end
        if abs(C2 - C(jj)) < opts.KKTViolationLevel * (C2 + C(jj) + opts.KKTViolationLevel)
            out = 0;
        else
            C1 = C(ii) + s * (C(jj) - C2);
            if C1 > 0 && C1 < C_max(ii)
                b1 = E1 + Y(ii).*(C1 - C(ii)).*kii + Y(jj).*(C2 - C(jj)).*kij + B
                B = b1;
            elseif C2 > 0 && C2 < C_max(jj)
                b2 = E2 + Y(ii).*(C1 - C(ii)).*kij + Y(jj).*(C2 - C(jj)).*kjj + B
                B = b2;
            else
                b1 = E1 + Y(ii).*(C1 - C(ii)).*kii + Y(jj).*(C2 - C(jj)).*kij + B
                b2 = E2 + Y(ii).*(C1 - C(ii)).*kij + Y(jj).*(C2 - C(jj)).*kjj + B
                B = (b1 + b2) / 2;
            end
            out = 1;
        end
        C(ii) = C1;
        C(jj) = C2;
    end
end
end

function[out] = examineExample(jj)
global m;
global X;
global Y;
global C;
global B;
global C_max;
global kernel;
global opts;

E2 = sum(C.*Y.*(kernel(X, X(jj,:)))) - B - Y(jj);
r2 = E2 * Y(jj);
if (r2 < -opts.TolKKT && C(jj) < C_max(jj)) || (r2 > opts.TolKKT && C(jj) > 0)
    num = sum(C > 0 & C < C_max);
    if num > 1%存在0<ai<C的元素个数
        MAX = 0;
        ii = 0;
        for kk = 1:m
            if C(kk) > 0 && C(kk) < C_max(kk)
                E1 = sum(C.*Y.*(kernel(X,X(kk,:)))) - B - Y(kk);
                if abs(E1 - E2) > MAX
                    MAX = abs(E1 - E2);
                    ii = kk;
                end
            end
        end
        out = takeStep(ii, jj);
    elseif num == 1
        for ii = 1:m
            if C(ii) > 0 && C(ii) < C_max(ii)
                out = takeStep(ii, jj);
            end
        end
    else
        ii = randi(m);
        out = takeStep(ii, jj);
    end
else
    out = 0;
end
end