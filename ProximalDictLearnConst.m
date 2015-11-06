function [ObjFunc, ObjFuncVal,A,X] = ProximalDictLearnConst(Y, lambda, beta, k, Ainit,MaxIter,Yv)
% Note: Yv is the validation data, size(Yv) = n x Nv
% Solve min 0.5*|Y-AX|_F^2 + lambda * |X|_1  s.t. |A|_F^2 <= beta
% Optimization variables are A,X
% X dimension k x N
% Y dimension n x N
% (c) Meisam Razaviyayn
%% initialization
[n N] = size(Y);
if nargin <= 4
    Ainit = randn(n,k);
    MaxIter = 300;
    Yv = zeros(n,1);
elseif nargin == 5
    MaxIter = 300;
    Yv = zeros(n,1);
elseif nargin == 6
    Yv = zeros(n,1);
else
end
Nv = size(Yv,2);
Xvinit = zeros(k,Nv);
NewtonEps = 0.01;
A = Ainit;
A = A / norm(A,'fro') * sqrt(beta);
ObjFunc = zeros(MaxIter,1);
ObjFuncVal = zeros(MaxIter,1);
relErrThe = 1e-6;   %stopping criterion
%% X initialization 
[X,obj_init,relerr_init] = L1L2MatNest(A,Y,lambda,(A' * A + lambda * eye(k))\(A' * Y),0,50);
XX = zeros(k,N);
%% begin iteration
for IterNum = 1: MaxIter
    % adaptive tau in every iteration
    opts.issym = 1;
    tau = 1 / eigs(A'*A,1,'lm',opts);
    ts = tic;
    XX = 0 * XX;
    temp = X - tau*A'*(A * X - Y);
    tempIndexPositive = find(temp>=lambda * tau);
    XX(tempIndexPositive) = temp(tempIndexPositive) - lambda *tau;
    tempIndexNegative = find(temp<= - lambda * tau);
    XX(tempIndexNegative) = temp(tempIndexNegative) + lambda *tau;    
    X = XX;
    TempXXt = X * X';
    TempYXt = Y * X';
    theta = 10^-11;   
    [U,Diag] = eig(TempXXt);
    eigVal = diag(Diag);
    TempYXtU = TempYXt*U;
    TempYXtUinvDiag = TempYXtU *  diag( 1 ./ (eigVal + theta)); 
    if (norm(TempYXtUinvDiag,'fro')^2) > beta       
        counter = 0;
        TempYXtUinvDiagNorm = norm(TempYXtUinvDiag,'fro');
        while (counter <= 20)&&not((TempYXtUinvDiagNorm^2 < beta*(1+NewtonEps))&&...
                (TempYXtUinvDiagNorm^2 > beta*(1-NewtonEps)))
            counter = counter + 1;
            constval = TempYXtUinvDiagNorm^2 - beta;
            consDerivative = - 2 * norm(TempYXtUinvDiag *  diag( 1 ./ (sqrt(eigVal + theta))),'fro')^2;
            theta = theta - constval/consDerivative;
            TempYXtUinvDiag = TempYXtU *  diag( 1 ./ (eigVal + theta)); 
            TempYXtUinvDiagNorm = norm(TempYXtUinvDiag,'fro');
        end
    end
    A = TempYXtUinvDiag * U';
    
    ObjFunc(IterNum) = (0.5 * norm(Y - A*X,'fro')^2 + lambda * sum(sum(abs(X))))/N;
    if (mod(IterNum,10)==1)&&(nargin>=5)   %calculate the validation set value every 10 iteration
        [Xvinit,obj_val,relerr_val] = L1L2MatNest(A,Yv,lambda,Xvinit);
        ObjFuncVal(IterNum) =(obj_val(end)) / Nv;
    end
    tElap = toc(ts);
    
    if IterNum > 1
        relerr = abs(ObjFunc(IterNum) - ObjFunc(IterNum-1)) / abs(ObjFunc(IterNum));
    else
        relerr = 1;
    end
    disp(strcat('Iter:', num2str(IterNum),', relerr:',num2str(relerr), ',time:', num2str(tElap),', train obj:', num2str(ObjFunc(IterNum)),', val obj:', num2str(ObjFuncVal(IterNum)) ) );
    if relerr < relErrThe
        break;
    end
end
ObjFunc = ObjFunc(1:IterNum);
ObjFuncVal = ObjFuncVal(1:IterNum);

end
