function [q,y,args] = optimizationKdV()
    
    ClearClose();

    args = CreateParameters();
    args.matrices = BuildMatrices(args);
    
%% Uncomment if you want to test one forward/backward simulation of a soliton    
%     args.kappa = 1.0;
%     args.x0 = -2.0;
%     args.xend = 5.0;
%     args.y0 = 12*args.kappa^2*sech(args.kappa*(args.chebyGL - args.x0)).^2;%valeurs aux chebypoints
%     q = 0.0*ones(args.nmax+2, args.N+1);
%     y = solveState(q,args);
%     plottedsteps=1:2:size(y.spatial,1);
%     [tg,xg] = meshgrid(args.tdata(plottedsteps),args.chebyGL(1:end));
%     set(gcf,'Position',[200,200,1500,1000])
%     surf(xg,tg,y.spatial(plottedsteps,:)');
%     p = solveAdjoint(q,y,args);
%     surf(xg,tg,p.spatial(plottedsteps,:)');


%% Uncomment if you want to test one forward simulation of a flow
%   encountering an isolated disturbance    
%    args.y0 = zeros(1,args.N+1);
%    q = 0.0*ones(args.nmax+2, args.N+1);
%    bumpvalue = 1.0;
%    bumpleft = args.N/2;
%    bumpright = args.N/2 + 5;
%    q(:,bumpleft) = bumpvalue;
%    q(:,bumpright) = -bumpvalue;
%    y = solveState(q,args);
%    plottedsteps=1:2:size(y.spatial,1);
%    [tg,xg] = meshgrid(args.tdata(plottedsteps),args.chebyGL(1:end));
%    set(gcf,'Position',[200,200,1500,1000])
%    surf(xg,tg,y.spatial(plottedsteps,:)');



%% Uncomment if you want to check gradient/hessian
    q = 1.0*ones(args.nmax+2, args.N+1);
    CheckGradient(q, q, @solveState, @solveAdjoint, ...
        @computeJ, @computeJp, args);
    CheckHessian(q, 100.0*q, @solveState, @solveAdjoint, ...
        @solveTangent, @solveDFH, @computeJ, @computeJpp, args);

%% Uncomment if goal is inverseproblem :find the source that created this
%% wave profile
     q = 0.0*ones(args.nmax+2, args.N+1);
     bumpvalue = 100.0;
     bumpleft = args.N/2;
     bumpright = args.N/2 + 5;
     q(:,bumpleft) = bumpvalue;
     q(:,bumpright) = -bumpvalue;
     args.y0 = zeros(1,args.N+1);
     y = solveState(q,args);
     args.dy0 = zeros(size(args.chebyGL));
     args.yobs = y.spatial;
     args.yspecobs = y.spec;

    
%%  Start of the continuation strategy
    fprintf('Continuation strategy...\n');
    
    q = zeros(args.nmax+2,args.N+1);%initialization of the control
    
    for i=1:size(args.gammaArray,2)
        gamma = args.gammaArray(i);% regularization term in 1/gamma
        fprintf('gamma = %d \n', gamma);
        update = 1;
        iter=0;

        y = solveState(q,args);% one forward simulation for y
        p = solveAdjoint(q,y,args);% one backward simulation for p
        
        L2NormInTimeP = sqrt(args.dt*sum((p.spatial).*(p.spatial)));
        for k=1:(args.N+1)    % check norm 0           
            if (L2NormInTimeP(k) <= args.epsilon)
                L2NormInTimeP(k) = args.alpha - args.epsilon;
            end
        end   
        ActiveSet = repmat(L2NormInTimeP > args.alpha,args.nmax+2,1);
        F = q - gamma/2*repmat(max(0,1-args.alpha./L2NormInTimeP),...
            args.nmax+2,1).*(p.spatial);
        normF = sqrt(sum(args.dt*sum(F.^2)))
        if (normF<args.tolNewton)
                update = 0;
        end

        while(update) %% Newton loop
            profile on;

            iter = iter+1;
            fprintf('Semi-smooth Newton method, gamma = %d, iteration %d\n',...
                gamma, iter);
            beta = 1;
            GradF = @(dq) ComputeOptCondGradient(q,dq,y,p,...
                        L2NormInTimeP,ActiveSet,gamma,...
                        @solveTangent,@solveDFH,args);

            [dq,success,residual,itermeth] = gmres(GradF,-F(:),[],10^-7);
            q = q + beta*reshape(dq,args.nmax+2,args.N+1);

            y = solveState(q,args);% one forward simulation for y
            p = solveAdjoint(q,y,args);% one backward simulation for p
            
            L2NormInTimeP = sqrt(args.dt*sum((p.spatial).*(p.spatial)));
            for k=1:(args.N+1)    % check norm 0           
                if (L2NormInTimeP(k) <= args.epsilon)
                    L2NormInTimeP(k) = args.alpha - args.epsilon;
                end
            end
            ActiveSet = repmat(L2NormInTimeP > args.alpha,args.nmax+2,1);
            F = q - gamma/2*repmat(max(0,1-args.alpha./L2NormInTimeP),...
                args.nmax+2,1).*(p.spatial);
            normFprev = normF;
            normF = sqrt(sum(args.dt*sum(F.^2)))
            if((normF < args.tolNewton) || (iter > args.iterNewton) || ...
                    abs(normF - normFprev) < 1e-4)
                update = 0;
            end
        end % end Newton loop
        myvisu(y.spatial,p.spatial,q,gamma,args);
    end % end gamma loop
end

function ClearClose()   
    % Close all figures including those with hidden handles
    close all hidden;

    % Store all the currently set breakpoints in a variable
    temporaryBreakpointData=dbstatus('-completenames');

    % Clear functions and their persistent variables (also clears breakpoints 
    % set in functions)
    clear functions;

    % Restore the previously set breakpoints
    dbstop(temporaryBreakpointData);

    % Clear global variables
    clear global;

    % Clear variables (including the temporary one used to store breakpoints)
    clear variables;
end

function args = CreateParameters()

    % Mesh
    args.D = 20; %domain is -50..50
    args.N = 160; %number of poinys
    args.k = args.N:-1:0;

    %Creation of Chebyshev Gauss-Lobatto points - our nodal basis
    args.chebyGL = cos(args.k*pi/args.N)*args.D;
    args.npoints = size(args.chebyGL,2);
    args.spacestep = [(args.chebyGL(2:end) - args.chebyGL(1:end-1))] ;

    %time argseters
    args.dt = 0.01;% time step for simulation
    args.tmax = 0.5;% maximum time for simulation
    args.nmax = round(args.tmax/args.dt);% induced number of time steps
    args.tdata = args.dt*(0:1:(args.nmax+1));

    % Optimization argseters
    args.alpha = 0.1;
    args.iterNewton = 20;
    args.tolNewton = 1e-5;
    args.epsilon = 1e-8;

    % Continuation strategy
    args.gammaArray = 2.^[1:20];
    %[10 100 500 1000 3000 7000 10000 30000 70000 100000 250000 500000 750000 1000000];

    % Misc
    args.coeffNL = 1.0;

    % Optimization stuff (for fsolve)
    args.optimOpt.TolFun = 1e-10;%1e-10 to have convenient gradient check
    args.optimOpt.Jacobian = 'on';
    args.optimOpt.DerivativeCheck = 'off';
    args.optimOpt.Display = 'off';
    %optimOpt.Algorithm = 'levenberg-marquardt';
    args.optimOpt.JacobMult = [];
    
    % default init
    args.y0 = zeros(1,args.N+1);
    args.dy0 = zeros(1,args.N+1);
    args.yobs = zeros(args.nmax+2,args.N+1);
    args.yspecobs = zeros(args.nmax+2,args.N-2);
    args.q = 0.0*ones(args.nmax+1, args.N+1);

end

function matrices = BuildMatrices(args)

    % Creation of Legendre polynomia
    LP = zeros(args.N+1, args.npoints);
    for i=0:args.N
    aux = legendre(i,args.chebyGL/args.D);
    LP(i+1,:) = aux(1,:);
    end

    % Creation of basis funtions for the dual petrov galerkin method - our spectral basis
    LP0 = LP(1:end-3,:);
    LP1 = LP(2:end-2,:);
    LP2 = LP(3:end-1,:);
    LP3 = LP(4:end,:);
    j = 0:(args.N-3);
    jm1 = 1:(args.N-2);
    jm2 = 2:(args.N-1);
    jm3 = 3:args.N;
    j1 = -1:(args.N-4);
    j2 = -2:(args.N-5);
    j3 = -3:(args.N-6);

    coeff = (2*j+3)./(2*j+5);
    diagcoeff = spdiags(coeff',0,args.N-2,args.N-2);
    LP1 = diagcoeff*LP1;
    LP3 = diagcoeff*LP3;
    trial = LP0 - LP1 - LP2 + LP3;
    test = LP0 + LP1 - LP2 - LP3;

    % Creation of the matrices

    % mass matrix
    mdiag0 = 2./(2*j+1) - 2*(2*j+3)./((2*j+5).^2) + 2./(2*j+5) - ...
        2*(2*j+3).^2./((2*j+7).*(2*j+5).*(2*j+5));
    mdiag1 = 6./(2*j1+7);
    mdiag2 = -2./(2*j2+5)+2*(2*j2+3)./((2*j2+5).*(2*j2+9));
    mdiag3 = -2*(2*j3+3)./((2*j3+5).*(2*j3+7));
    mdiagm1 = -6./(2*jm1+5);
    mdiagm2 = -2./(2*jm2+1) + 2*(2*jm2-1)./((2*jm2+1).*(2*jm2+5));
    mdiagm3 = 2*(2*jm3-3)./((2*jm3-1).*(2*jm3+1));
    M =args.D*spdiags([mdiagm3' mdiagm2' mdiagm1' mdiag0' mdiag1' mdiag2' mdiag3'],...
        -3:3, args.N-2, args.N-2);

    % matrix for linear transport term 
    pdiag0 = 4*(2*j+3)./(2*j+5);
    pdiag1 = -8./(2*j1+7);
    pdiag2 = -2*(2*j2+3)./(2*j2+5);
    pdiagm1 = 8./(2*jm1+5);
    pdiagm2 = -2*(2*jm2-1)./(2*jm2+1);
    P = spdiags([pdiagm2' pdiagm1' pdiag0' pdiag1' pdiag2'],...
        -2:2, args.N-2, args.N-2);

    % matrix for 3rd order term
    sdiag0 =2*(2*j+3).^2;
    S = 1.0/(args.D^2)*spdiags(sdiag0',0,args.N-2,args.N-2);

    % mass matrix for trial basis only (to compute adjoint)
    adiag0 = -2./(2*j+5) + 8./(2*j+7) + 2./(2*j+1);
    adiag1 = -2./(2*j1+5) + 2./(2*j1+7) - 2*(2*j1+3)./((2*j1+5).*(2*j1+7));
    adiag2 = -2./(2*j2+5) - 2*(2*j2+3)./((2*j2+5).*(2*j2+9));
    adiag3 = 2*(2*j3+3)./((2*j3+5).*(2*j3+7));
    adiagm1 = -2./(2*j+5) + 2./(2*j+7) - 2*(2*j+3)./((2*j+5).*(2*j+7));
    adiagm2 = -2./(2*j+5) - 2*(2*j+3)./((2*j+5).*(2*j+9));
    adiagm3 = 2*(2*j+3)./((2*j+5).*(2*j+7));
    A = args.D*spdiags([adiagm3' adiagm2' adiagm1' adiag0' adiag1' adiag2' adiag3'], ...
        -3:3, args.N-2, args.N-2);

    % Fill in the structure
    matrices.A = A;
    matrices.M = M;
    matrices.MTInv = inv(matrices.M');
    matrices.S = S;
    matrices.P = P;
    matrices.test = test;
    matrices.trial = trial;
    matrices.testT = test';
    matrices.trialT = trial';
    matrices.Imp = matrices.M - 0.5*args.dt*matrices.P + 0.5*args.dt*matrices.S;
    matrices.Exp = matrices.M + 0.5*args.dt*matrices.P - 0.5*args.dt*matrices.S;
    matrices.ImpT = matrices.M' - 0.5*args.dt*matrices.P' + 0.5*args.dt*matrices.S';
    matrices.ExpT = matrices.M' + 0.5*args.dt*matrices.P' - 0.5*args.dt*matrices.S';
    matrices.trialTInv = pinv(matrices.trialT);
    matrices.Pdt = 0.5*args.dt*0.5*matrices.P;
    matrices.trialTInvTPT = (matrices.trialTInv')*(matrices.P');
    matrices.solverexp=args.coeffNL*0.5*args.dt*0.5*matrices.P*matrices.trialTInv;
    matrices.solver=args.coeffNL*matrices.Pdt*matrices.trialTInv;
    matrices.solvertangent = args.coeffNL*0.5*args.dt*matrices.P*matrices.trialTInv;
    matrices.solvertangentadjoint = args.coeffNL*0.5*args.dt*matrices.trial;
    matrices.sizeTA = size(matrices.trialTInvTPT,2);
    matrices.sizeT = size(matrices.trialT,2);

end

%% %%%%%%%%%%%%%%%% solveState functions %%%%%%%%%%%%%%%%
function y = solveState(q, args)

    nmax=args.nmax;
    N=args.N;
    matrices = args.matrices;

    yspec0 = matrices.trialT\(args.y0)';

    qspec = matrices.trialT\q';%zeros(nmax+1, N-2);
    qspec=qspec';

    y.spatial = zeros(nmax+2,N+1);
    y.spec = zeros(nmax+2,N-2);
    y.spatial(1,:) = args.y0;
    y.spec(1,:) = yspec0;
    %% Time loop
    yspecm1 = yspec0;
    for i=2:nmax+2
        b = explicitpart(yspecm1,qspec(i,:)', qspec(i-1,:)',args);
        yspeci = fsolve(@(x) fsolverFun(x,b,args),yspecm1,args.optimOpt);
        yi = matrices.trialT*yspeci;
        yspecm1 = yspeci;
        y.spec(i,:) = yspeci';
        y.spatial(i,:) = yi';
    end
end

function f = explicitpart(y,qn,qnm1,args)
    matrices = args.matrices;
    f = matrices.Exp*y...
       + matrices.solverexp*((matrices.trialT*y).^2)...
       + 0.5*args.dt*(matrices.M)*qn + 0.5*args.dt*(matrices.M)*qnm1;
end

function [F,J] = fsolverFun(u, b, args)
    matrices = args.matrices;
    Tu = matrices.trialT*u;
    F = matrices.Imp*u - matrices.solver*(Tu.^2) - b;
    J = matrices.Imp - 2.0*matrices.solver*(repmat(Tu, 1, size(matrices.trialT,2)).*matrices.trialT);
end

%% %%%%%%%%%%%%%%%% solveAdjoint functions %%%%%%%%%%%%%%%%
function p = solveAdjoint(q,y,args)
    dt=args.dt;
    nmax=args.nmax;
    N=args.N;
    matrices = args.matrices;
    
    yrev = y.spatial(end:-1:1,:);
    yspecrev = y.spec(end:-1:1,:);
    
    % source term
    yspecobsrev = args.yspecobs(end:-1:1,:);

    %first step
    b = -0.5*args.dt*matrices.A*(y.spec(end,:)' - args.yspecobs(end,:)');
    pspec0 = fsolve(@(x) fsolverFunAdjoint(x,yspecrev(1,:)',b,args),...
        y.spec(end,:)',args.optimOpt);
    p0 = matrices.testT*pspec0;
    p.spatial = zeros(nmax+2,N+1);
    p.spec = zeros(nmax+2,N-2);
    p.spatial(1,:) = p0;
    p.spec(1,:) = pspec0;

    %% Time loop
    pspec1 = pspec0;
    for i=2:nmax+1
        b = explicitpartAdjoint(pspec1,yspecrev(i,:)',args) ...
            - args.dt*matrices.A*(yspecrev(i,:)' - yspecobsrev(i,:)');
        pspec = fsolve(@(x) fsolverFunAdjoint(x,yspecrev(i,:)',b,...
           args),pspec1,args.optimOpt);
        pp = matrices.testT*pspec;
        pspec1 = pspec;
        p.spec(i,:) = pspec';
        p.spatial(i,:) = pp';
    end
     pspecend = matrices.MTInv*(matrices.ExpT*p.spec(nmax+1,:)' +...
         0.5*dt*matrices.trial*((matrices.trialT*y.spec(1,:)').*...
         (matrices.trialTInvTPT*p.spec(nmax+1,:)'))...
         - 0.5*args.dt*matrices.A*(y.spec(1,:)'- args.yspecobs(1,:)'));
     pend = matrices.testT*pspecend;
     p.spec(end,:) = pspecend';
     p.spatial(end,:) = pend';
 
     p.spec = p.spec(end:-1:1,:);
     p.spatial = p.spatial(end:-1:1,:);
end

function f = explicitpartAdjoint(p,u,args)
    matrices = args.matrices;
    Tu = matrices.trialT*u;
    Tp = matrices.trialTInvTPT*p;
    f = matrices.ExpT*p + args.coeffNL*0.5*args.dt*matrices.trial*...
        (Tu.*Tp);
end

function [F,J] = fsolverFunAdjoint(p,y,b,args)
    matrices = args.matrices;
    Ty = matrices.trialT*y;
    Tp = matrices.trialTInvTPT*p;
    F = matrices.ImpT*p - args.coeffNL*0.5*args.dt*matrices.trial*(Ty.*Tp) - b;
    J = matrices.ImpT - args.coeffNL*0.5*args.dt*matrices.trial*...
        (repmat(Ty,1,size(matrices.trialTInvTPT,2)).*matrices.trialTInvTPT);
end

%% %%%%%%%%%%%%%%%% CheckGradient functions %%%%%%%%%%%%%%%%
function j = computeJ(q,y,args)
    j = 0.0;
    init=1;
    discr = y.spec(init,:) - args.yspecobs(init,:);
    j = j + 0.5*args.dt*0.5*discr*(args.matrices.A)*discr';
    for i=2:args.nmax+1
        discr = y.spec(i,:) - args.yspecobs(i,:);
        j = j + args.dt*0.5*discr*(args.matrices.A)*discr';
    end
    nend = args.nmax+2;
    discr = y.spec(nend,:) - args.yspecobs(nend,:);
    j = j + 0.5*args.dt*0.5*discr*(args.matrices.A)*discr';
end

function g = computeJp(dq,q,y,p,args) %signe comme dans la these de D.Meidner
    nmax = args.nmax;
    dt = args.dt;
    M = args.matrices.M;
    g = 0.0;
    for i=2:nmax+2
       pspeci = p.spec(i,:);
       q1 = dq(i,:);
       q2 = dq(i-1,:);
       qspec1 = args.matrices.trialT\q1';
       qspec2 = args.matrices.trialT\q2';
       g = g - 0.5*dt*pspeci*M*qspec1 - 0.5*dt*pspeci*M*qspec2;
    end
end

%% %%%%%%%%%%%%%%%% solveTangent functions %%%%%%%%%%%%%%%%
function dy = solveTangent(q, y, dq, args)
    nmax = args.nmax;
    N = args.N;
    matrices = args.matrices;
    
    dyspec0 = matrices.trialT\args.dy0';
    dqspec = matrices.trialT\dq';
    dqspec=dqspec';

    dy.spatial = zeros(nmax+2,N+1);
    dy.spec = zeros(nmax+2,N-2);
    dy.spatial(1,:) = args.dy0;
    dy.spec(1,:) = dyspec0;
    %% Time loop
    dyspecm1 = dyspec0;
        for i=2:nmax+2
            b = explicitpartTangent(dyspecm1,y.spec(i-1,:)',...
                dqspec(i,:)',dqspec(i-1,:)',args);
            dyspeci = fsolve(@(x) fsolverFunTangent(y.spec(i,:)',x,b,args)...
                ,dyspecm1,args.optimOpt);
            dyi = matrices.trialT*dyspeci;
            dyspecm1 = dyspeci;
            dy.spec(i,:) = dyspeci;
            dy.spatial(i,:) = dyi;
        end
end

function f = explicitpartTangent(dyspec,yspec,dqn,dqnp1,args)
    matrices = args.matrices;
    dt = args.dt;
    f = matrices.Exp*dyspec...
       + args.coeffNL*0.5*dt*matrices.P*...
       (matrices.trialTInv*((matrices.trialT*yspec).*...
       (matrices.trialT*dyspec)))...
       + 0.5*dt*(matrices.M)*dqn + 0.5*dt*(matrices.M)*dqnp1;
end

function [F,J] = fsolverFunTangent(yspec,dy,b,args)
    matrices = args.matrices;
    Ty = matrices.trialT*yspec;
    Tdy = matrices.trialT*dy;
    F = matrices.Imp*dy - matrices.solvertangent*(Ty.*Tdy) - b;
    J = matrices.Imp - matrices.solvertangent*...
        (repmat(Ty,1,matrices.sizeT).*matrices.trialT);
end

%% %%%%%%%%%%%%%%%% solveDFH functions %%%%%%%%%%%%%%%%
function dp = solveDFH(q, y, p, dq, dy, args)
    dt=args.dt;
    nmax=args.nmax;
    N=args.N;
    matrices = args.matrices;

    yspecrev = y.spec(end:-1:1,:);
    dyspecrev = dy.spec(end:-1:1,:);
    pspecrev = p.spec(end:-1:1,:);
    
    yspecobsrev = args.yspecobs(end:-1:1,:);

    %first step
    b = -0.5*args.dt*matrices.A*(dy.spec(end,:)')...
        +args.coeffNL*0.5*dt*matrices.trial*...
        (matrices.trialT*dyspecrev(1,:)'.*...
        (matrices.trialTInvTPT*pspecrev(1,:)'));
    dpspec0 = fsolve(@(x) fsolverFunDFH(x,yspecrev(1,:)',b,args),...
        zeros(args.N-2,1),args.optimOpt);
    dp0 = matrices.testT*dpspec0;
    dp.spatial = zeros(nmax+2,N+1);
    dp.spec = zeros(nmax+2,N-2);
    dp.spatial(1,:) = dp0;
    dp.spec(1,:) = dpspec0;

    %% Time loop
    dpspecm1 = dpspec0;
      for i=2:nmax+1
       b = explicitpartDFH(dpspecm1,pspecrev(i-1,:)',...
           pspecrev(i,:)', ...
           dyspecrev(i,:)',yspecrev(i,:)',args)...
           - args.dt*matrices.A*(dyspecrev(i,:)');
       dpspeci = fsolve(@(x) fsolverFunDFH(x,yspecrev(i,:)',b,args),...
           dpspecm1,args.optimOpt);
       dpi = matrices.testT*dpspeci;
       dpspecm1 = dpspeci;
       dp.spec(i,:) = dpspeci;
       dp.spatial(i,:) = dpi;
      end

    %last step
    dpspecend = matrices.MTInv*(  matrices.ExpT*dp.spec(nmax+1,:)' +...
        args.coeffNL*0.5*dt*matrices.trial*(...
          (matrices.trialT*yspecrev(end,:)').*...
          (matrices.trialTInvTPT*dp.spec(nmax+1,:)')...
        + (matrices.trialT*dyspecrev(end,:)').*...
        (matrices.trialTInvTPT*pspecrev(nmax+1,:)')...
                                )...   
        - 0.5*args.dt*matrices.A*(dy.spec(1,:)'));
    dpend = matrices.testT*dpspecend;
    dp.spec(end,:) = dpspecend;
    dp.spatial(end,:) = dpend;

    dp.spec = dp.spec(end:-1:1,:);
    dp.spatial = dp.spatial(end:-1:1,:);

end

function f = explicitpartDFH(dp,pexp,pimp,dy,y,args)
    matrices = args.matrices;
    Tdy = matrices.trialT*dy;
    Tdp = matrices.trialTInvTPT*dp;
    Ty = matrices.trialT*y;
    Tpexp = matrices.trialTInvTPT*pexp;
    Tpimp = matrices.trialTInvTPT*pimp;
    nltermexp = args.coeffNL*0.5*args.dt*matrices.trial*...
        (Tdy.*Tpexp + Ty.*Tdp);
    nltermimp = args.coeffNL*0.5*args.dt*matrices.trial*(Tdy.*Tpimp);
    f = matrices.ExpT*dp + nltermexp + nltermimp;
end

function [F,J] = fsolverFunDFH(dp,y,b,args)
    matrices = args.matrices;
    Ty = matrices.trialT*y;
    Tdp = matrices.trialTInvTPT*dp;
    F = matrices.ImpT*dp - matrices.solvertangentadjoint*(Ty.*Tdp) - b;
    J = matrices.ImpT - matrices.solvertangentadjoint*...
        (repmat(Ty,1,matrices.sizeTA).*matrices.trialTInvTPT);
end

%% %%%%%%%%%%%%%%%% CheckHessian functions %%%%%%%%%%%%%%%%
function h = computeJpp(q, y, p, dq, dy, dp, args)
    nmax=args.nmax;
    dt=args.dt;
    matrices = args.matrices;
    M = matrices.M;
    h=0.0;
    for i=2:nmax+2
        dpspeci = dp.spec(i,:);
        dq1 = dq(i,:);
        dq2 = dq(i-1,:);
        dqspec1 = matrices.trialT\dq1';
        dqspec2 = matrices.trialT\dq2';
        h = h - 0.5*dt*dpspeci*M*dqspec1 - 0.5*dt*dpspeci*M*dqspec2;
    end
end

%% %%%%%%%%%%%%%%%% Misc %%%%%%%%%%%%%%%%
function GradF = ComputeOptCondGradient(q,dq,y,p,...
                        L2NormInTimeP,ActiveSet,gamma,...
                        solveTangent,solveDFH,args)
    nmax=args.nmax;
    N=args.N;
    dt=args.dt;
    dq = reshape(dq,nmax+2,N+1);
    dy = solveTangent(q,y,dq,args);
    dp = solveDFH(q,y,p,dq,dy,args);

    grad1 = repmat(max(0,1-args.alpha./L2NormInTimeP),nmax+2,1).*...
        (dp.spatial);
    grad2 = args.alpha*ActiveSet.*...
        repmat(dt*sum((p.spatial).*(dp.spatial))./(L2NormInTimeP.^3),...
        nmax+2,1).*(p.spatial);
    GradF = dq - 0.5*gamma*(grad1 + grad2);
    GradF = GradF(:);
end


function myvisu(y,p,q,gamma,args,plottedsteps)
    %% 3D - Vizualization
    plottedsteps=1:2:size(y,1);
    [tg,xg] = meshgrid(args.tdata(plottedsteps),args.chebyGL(1:end));
    
    subplot(2,2,1), surf(xg,tg,y(plottedsteps,:)');
    xlabel('x');ylabel('Time');zlabel('State variable y');
    title('State Variable y');
    %axis([-16,16,0,0.5,-1.5,1.5]);
    view(-16,10);
    shading interp;
    
    subplot(2,2,2), surf(xg,tg,p(plottedsteps,:)');
    xlabel('x');ylabel('Time');zlabel('Adjoint variable p');
    title('Adjoint state');
    %axis([-16,16,0,0.5,-0.1,0.1]);
    view(-16,10);
    shading interp;
    
    subplot(2,2,3), surf(xg,tg,q(plottedsteps,:)');
    xlabel('x');ylabel('Time');zlabel('Control variable q');
    title('Current Control');
    %axis([-16,16,0,0.5,-2,3]);
    view(-16,10);
    shading interp;
    
    subplot(2,2,4), surf(xg,tg,y(plottedsteps,:)'-args.yobs(plottedsteps,:)');
    xlabel('x');zlabel('Y - Yobs');
    title('Error');
    view(-16,10);
    shading interp;

    str = sprintf('Continuation strategy, gamma = %d', gamma);
    suptitle(str);
end

function exportPgfplot3d(filename,xg,tg,var,gamma,args)
    fichier=fopen(strcat(filename,num2str(gamma),'.txt'),'w+');
    fmt=['%6.4f %6.4f %6.4f\n'];
    var2=var(plottedsteps,:)';
    n=size(Y2,1);
    for l=1:size(Y2,2)
        x=xg(:,l);
        t=tg(:,l);
        vec=var2(:,l);
        fprintf(fichier,fmt,[x t vec]');
        fprintf(fichier,'\n');
    end
    fclose(fichier);
end

function exportPgfplot2d(filename,x,var,gamma,args)
    fichier=fopen(strcat(filename,num2str(gamma),'.txt'),'w+');    
    fmt=['%6.4f %6.4f\n'];
    fprintf(fichier,fmt,[x var]');
    fprintf(fichier,'\n');
    fclose(fichier);
end