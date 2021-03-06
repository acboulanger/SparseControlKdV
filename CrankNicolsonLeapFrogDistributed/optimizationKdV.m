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
%     subplot(1,2,1);
%     surf(xg,tg,y.spatial(plottedsteps,:)');
%     args.yobs = zeros(size(y.spatial));
%     args.yspecobs = zeros(size(y.spec)); 
%     p = solveAdjoint(q,y,args);
%     subplot(1,2,2);
%     plottedsteps2=1:2:size(p.spatial,1);
%     [tg2,xg2] = meshgrid(args.tdata(plottedsteps2),args.chebyGL(1:end));
%     surf(xg2,tg2,p.spatial(plottedsteps2,:)');


%% Uncomment if you want to test one forward simulation of a flow
 % encountering an isolated disturbance    
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
%    q = 1.0*ones(args.nmax+1, args.N+1);
%    CheckGradient(q, q, @solveState, @solveAdjoint, ...
%        @computeJ, @computeJp, args);
%    CheckHessian(q, q, @solveState, @solveAdjoint, ...
%        @solveTangent, @solveDFH, @computeJ, @computeJpp, args);

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
% 
% 
%     
%%  Start of the continuation strategy
    fprintf('Continuation strategy...\n');
    
    q = zeros(args.nmax+1,args.N+1);%initialization of the control
    
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
        ActiveSet = repmat(L2NormInTimeP > args.alpha,args.nmax+1,1);
        F = q - gamma/2*repmat(max(0,1-args.alpha./L2NormInTimeP),...
            args.nmax+1,1).*(p.spatial);
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
            q = q + beta*reshape(dq,args.nmax+1,args.N+1);

            y = solveState(q,args);% one forward simulation for y
            p = solveAdjoint(q,y,args);% one backward simulation for p
            
            L2NormInTimeP = sqrt(args.dt*sum((p.spatial).*(p.spatial)));
            for k=1:(args.N+1)    % check norm 0           
                if (L2NormInTimeP(k) <= args.epsilon)
                    L2NormInTimeP(k) = args.alpha - args.epsilon;
                end
            end
            ActiveSet = repmat(L2NormInTimeP > args.alpha,args.nmax+1,1);
            F = q - gamma/2*repmat(max(0,1-args.alpha./L2NormInTimeP),...
                args.nmax+1,1).*(p.spatial);
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
    args.N = 160; %number of points
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
    args.tolgmres = 1e-8;

    % Continuation strategy
    args.gammaArray = 2.^[1:20];
    %[10 100 500 1000 3000 7000 10000 30000 70000 100000 250000 500000 750000 1000000];

    % Misc
    args.coeffNL = 1.0;

    % Optimization stuff (for fsolve)
    args.optimOpt.TolFun = 1e-5;
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
    M =args.D*spdiags([mdiagm3' mdiagm2' mdiagm1' mdiag0'...
        mdiag1' mdiag2' mdiag3'],...
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
    matrices.MT = M';
    matrices.MTInv = inv(M');
    matrices.S = S;
    matrices.P = P;
    matrices.PT = P';
    matrices.test = test;
    matrices.trial = trial;
    matrices.trialT = trial';
    matrices.testT = test';
    matrices.testTInv = pinv(test');
    matrices.trialT = trial';
    matrices.trialTInv = pinv(trial');
    matrices.left=0.5*(M +args.dt*S);
    matrices.leftInv = inv(matrices.left);
    matrices.leftTInv = inv(matrices.left');
    matrices.right=0.5*(M -args.dt*S);
    matrices.rightT=matrices.right';
    matrices.M_leftTinv_dt = matrices.leftTInv*args.dt;
    matrices.M_leftTinv_rightT = matrices.leftTInv*(matrices.rightT);

end

%% %%%%%%%%%%%%%%%% solveState functions %%%%%%%%%%%%%%%%
function y = solveState(q, args)
    % parameters
    dt=args.dt;
    nmax=args.nmax;
    N=args.N;
    coeffNL=args.coeffNL;
    matrices = args.matrices;
    
    % state variables
    y.spatial = zeros(nmax+2,N+1);% spatial domain
    y.spec = zeros(nmax+2,N-2);% spectral domain
    
    % initialization
    yspec0 = matrices.trialT\(args.y0)';
    y.spatial(1,:) = args.y0;
    y.spec(1,:) = yspec0; 
    qspec = matrices.trialT\q';
    qspec=qspec';

    % first time step in the spectral space, semi implicit
    NLterm = (args.y0).^2;
    pNLterm=coeffNL*matrices.trialTInv*NLterm';
    yspec1 = matrices.leftInv*((0.5*matrices.M)*yspec0 ...
            + 0.5*dt*(0.5*matrices.P*pNLterm + matrices.P*yspec0...
            + matrices.M*qspec(1,:)'));
    y1 = matrices.trialT*yspec1;
    y.spatial(2,:) = y1;
    y.spec(2,:) = yspec1;

    % Time loop for interior steps - Crank Nicolson Leap Frog scheme
    ym1 = y1;
    yspecm1 = yspec1;
    yspecm2 = yspec0;
    for i=2:nmax
        NLterm = ym1.*ym1;
        pNLterm=coeffNL*matrices.trialTInv*NLterm;
        yspeci = (matrices.leftInv)*( (matrices.right)*yspecm2 ...
          + dt*(0.5*matrices.P*pNLterm...
          + matrices.P*yspecm1 ...
          + matrices.M*qspec(i,:)') );
        yi = matrices.trialT*yspeci;
        ym1 = yi;
        yspecm2 = yspecm1;
        yspecm1 = yspeci;
        y.spec(i+1,:) = yspeci;
        y.spatial(i+1,:) = yi;
    end

    %last step 
    %minv=inv(matrices.M);
    NLterm = ym1.^2;
    pNLterm=coeffNL*matrices.trialTInv*NLterm;
    
%      rhs = ((matrices.right)*yspecm2 +...
%                  0.5*dt*(0.5*matrices.P*pNLterm + matrices.P*yspecm1)...
%                  + 0.5*matrices.M*yspecm1...
%                  + 1.0*0.5*dt*matrices.M*qspec(nmax+1,:)');
%      [yspecend,success,residual,itermeth] = gmres(matrices.M,rhs,[],args.tolgmres,N-2);
     yspecend = (matrices.M)\((matrices.right)*yspecm2 +...
                 0.5*dt*(0.5*matrices.P*pNLterm + matrices.P*yspecm1)...
                 + 0.5*matrices.M*yspecm1...
                 + 1.0*0.5*dt*matrices.M*qspec(nmax+1,:)');
    yend = matrices.trialT*yspecend;
    y.spec(end,:) = yspecend;
    y.spatial(end,:) = yend;
end


%% %%%%%%%%%%%%%%%% solveAdjoint functions %%%%%%%%%%%%%%%%
function p = solveAdjoint(q,y,args)
    % parameters
    dt=args.dt;
    nmax=args.nmax;
    N=args.N;
    coeffNL = args.coeffNL;
    matrices = args.matrices;
    
    % state variables
    p.spatial = zeros(nmax+1,N+1);
    p.spec = zeros(nmax+1,N-2);

    yrev = y.spatial(end:-1:1,:);
    yspecrev = y.spec(end:-1:1,:);
    
    % source term
    yspecobsrev = args.yspecobs(end:-1:1,:);
    
    % initialization 
    % M is ill conditionned. One solve the projection problem
    % by minimal residuals
    mt = matrices.MT;
%     rhs = -dt*matrices.A*(yspecrev(1,:)' - yspecobsrev(1,:)');
%     [pspec0,success,residual,itermeth] = gmres(mt,rhs,[],args.tolgmres,N-2);
    pspec0 = - mt\(dt*matrices.A'*(yspecrev(1,:)' - yspecobsrev(1,:)'));
    p0 = (matrices.testT)*(pspec0);
    p.spatial(1,:) = p0;
    p.spec(1,:) = pspec0;

    % first step
    NLterm = 0.5*2.0*yrev(2,:)'.*(matrices.trialTInv'*(matrices.PT*pspec0));
    pNLterm=coeffNL*matrices.trial*NLterm;
    pspec1 = matrices.leftTInv*(0.5*matrices.MT*pspec0 +...
        0.5*dt*(matrices.PT*pspec0 + pNLterm ...
        - 2.0*matrices.A*(yspecrev(2,:)' - yspecobsrev(2,:)')));
    p1 = matrices.testT*pspec1;
    p.spatial(2,:) = p1;
    p.spec(2,:) = pspec1;
    
    pspecm1 = pspec1;
    pspecm2 = pspec0;
    %Time loop
    for i = 2:nmax
        NLterm = 0.5*2.0*yrev(i+1,:)'.*(matrices.trialTInv'*(matrices.PT*pspecm1));
        pNLterm=coeffNL*matrices.trial*NLterm;
        pspeci = matrices.M_leftTinv_rightT*pspecm2...
            + matrices.M_leftTinv_dt* (matrices.PT*pspecm1 + pNLterm ...
            - matrices.A*(yspecrev(i+1,:)' - yspecobsrev(i+1,:)'));
        pi = matrices.testT*pspeci;
        pspecm2 = pspecm1;
        pspecm1 = pspeci; 
        p.spec(i+1,:) = pspeci;
        p.spatial(i+1,:) = pi;
    end
    p.spec = p.spec(end:-1:1,:);
    p.spatial = p.spatial(end:-1:1,:);
end


%% %%%%%%%%%%%%%%%% CheckGradient functions %%%%%%%%%%%%%%%%
function j = computeJ(q,y,args)
    j = 0.0;
    for i=2:args.nmax+2
        discr = y.spec(i,:) - args.yspecobs(i,:);
        j = j + args.dt*0.5*discr*(args.matrices.A)*discr';
    end
end

function g = computeJp(dq,q,y,p,args)
    nmax = args.nmax;
    dt = args.dt;
    matrices = args.matrices;
    M = matrices.M;
    dq1 = matrices.trialT\dq(1,:)';
    g = -0.5*dt*p.spec(1,:)*M*dq1;
    
    for i=2:nmax
        pi = p.spec(i,:);
        dqi = dq(i,:);
        dqspeci = matrices.trialT\dqi';
        g = g - dt*pi*M*dqspeci;
    end
    dqend =  matrices.trialT\dq(nmax+1,:)';
    g = g -0.5*dt*p.spec(nmax+1,:)*M*dqend;
end

%% %%%%%%%%%%%%%%%% solveTangent functions %%%%%%%%%%%%%%%%
function dy = solveTangent(q, y, dq, args)
    dt=args.dt;
    nmax=args.nmax;
    N=args.N;
    coeffNL = args.coeffNL;
    matrices = args.matrices;

    % Storage matrices
    dy.spatial = zeros(nmax+2,N+1);
    dy.spec = zeros(nmax+2,N-2);

    % initial condition and source in the spectral space
    dyspec0=matrices.trialTInv*(args.dy0)';
    dqspec = matrices.trialTInv*dq';
    dqspec=dqspec';
    dy.spatial(1,:) = args.dy0;
    dy.spec(1,:) = dyspec0;

    %first time step in the spectral space, semi implicit
    NLterm = y.spatial(1,:).*(args.dy0);
    pNLterm = coeffNL*matrices.trialTInv*NLterm';
    dyspec1 = matrices.leftInv*((0.5*matrices.M)*dyspec0 ...
        + 0.5*dt*(matrices.P*pNLterm + matrices.P*dyspec0 ...
        + matrices.M*dqspec(1,:)'));
    dy1 = matrices.trialT*dyspec1;
    dy.spatial(2,:) = dy1;
    dy.spec(2,:) = dyspec1;

    % Time loop
    dym1 = dy1;
    dyspecm1 = dyspec1;
    dyspecm2 = dyspec0;
    for i=2:nmax
        NLterm = y.spatial(i,:)'.*dym1;
        pNLterm = coeffNL*matrices.trialTInv*NLterm;
        dyspeci = (matrices.leftInv)*( (matrices.right)*dyspecm2 ...
          + dt*(matrices.P*pNLterm...
          + matrices.P*dyspecm1 ...
          + matrices.M*dqspec(i,:)') );
        dyi = matrices.trialT*dyspeci;
        dym1 = dyi;
        dyspecm2 = dyspecm1;
        dyspecm1 = dyspeci;
        dy.spec(i+1,:) = dyspeci;
        dy.spatial(i+1,:) = dyi;
    end
    
    % last step
    minv=inv(matrices.M);
    NLterm = y.spatial(nmax+1,:)'.*dym1;
    pNLterm=coeffNL*matrices.trialTInv*NLterm; 
    
%       rhs = ((matrices.right)*dyspecm2...
%                       + 0.5*dt*(matrices.P*pNLterm + matrices.P*dyspecm1)...
%                       + 0.5*matrices.M*dyspecm1...
%                       + 1.0*0.5*dt*matrices.M*dqspec(nmax+1,:)');
%       [dyspecend,success,residual,itermeth] = gmres(matrices.M,rhs,[],args.tolgmres,N-2);
%     
     dyspecend = (matrices.M)\((matrices.right)*dyspecm2...
                     + 0.5*dt*(matrices.P*pNLterm + matrices.P*dyspecm1)...
                     + 0.5*matrices.M*dyspecm1...
                     + 1.0*0.5*dt*matrices.M*dqspec(nmax+1,:)');
     dyend = matrices.trialT*dyspecend;
     dy.spec(end,:) = dyspecend;
     dy.spatial(end,:) = dyend;
end


%% %%%%%%%%%%%%%%%% solveDFH functions %%%%%%%%%%%%%%%%
function dp = solveDFH(q, y, p, dq, dy, args)
    dt=args.dt;
    nmax=args.nmax;
    N=args.N;
    coeffNL = args.coeffNL;
    matrices = args.matrices;

    %source term
    yrev = y.spatial(end:-1:1,:);
    dyrev = dy.spatial(end:-1:1,:);
    pspecrev=p.spec(end:-1:1,:);
    dyspecrev = dy.spec(end:-1:1,:);

    % Storage array
    dp.spatial = zeros(nmax+1,N+1);
    dp.spec = zeros(nmax+1,N-2);

    %initial condition
    mt = matrices.MT;
    %rhs = -dt*matrices.A*(dyspecrev(1,:)');
    %[dpspec0,success,residual,itermeth] = gmres(mt,rhs,[],args.tolgmres,N-2);
    dpspec0= mt\(dt*matrices.A*(-dyspecrev(1,:)'));
    dp0 = (matrices.testT)*(dpspec0);
    dp.spatial(1,:) = dp0;
    dp.spec(1,:) = dpspec0;

    %first step

    NLterm = yrev(2,:)'.*(matrices.trialTInv'*(matrices.PT*dpspec0));
    NLterm2 = dyrev(2,:)'.*...
        (matrices.trialTInv'*(matrices.PT*pspecrev(1,:)'));
    pNLterm = coeffNL*matrices.trial*(NLterm+NLterm2);

    dpspec1 = matrices.leftTInv*(0.5*matrices.MT*dpspec0 + ...
        0.5*dt*(matrices.PT*dpspec0 + pNLterm) - dt*matrices.A*(dyspecrev(2,:)'));
    dp1 = matrices.testT*dpspec1;

    dp.spatial(2,:) = dp1;
    dp.spec(2,:) = dpspec1;
    dpspec2 = dpspec0;
    
    % Time loop
    for i = 2:(nmax)
        NLterm = yrev(i+1,:)'.*(matrices.trialTInv'*(matrices.PT*dpspec1));
        NLterm2 = dyrev(i+1,:)'.*...
            (matrices.trialTInv'*(matrices.PT*pspecrev(i,:)'));
        pNLterm = coeffNL*matrices.trial*(NLterm+NLterm2);

        dpspeci=matrices.M_leftTinv_rightT*dpspec2...
                + matrices.M_leftTinv_dt* (matrices.PT*dpspec1 + pNLterm...
                - matrices.A*(dyspecrev(i+1,:)'));
        dpi = matrices.testT*dpspeci;
        dpspec2 = dpspec1;
        dpspec1 = dpspeci; 
        dp.spec(i+1,:) = dpspeci;
        dp.spatial(i+1,:) = dpi;
    end
    %reverse results
    dp.spec = dp.spec(end:-1:1,:);
    dp.spatial = dp.spatial(end:-1:1,:);
end


%% %%%%%%%%%%%%%%%% CheckHessian functions %%%%%%%%%%%%%%%%
function h = computeJpp(q, y, p, dq, dy, dp, args)
    nmax=args.nmax;
    dt=args.dt;
    matrices = args.matrices;
    M = matrices.M;
    
    dq1 = matrices.trialT\dq(1,:)';
    h = -0.5*dt*dp.spec(1,:)*M*dq1;
    for i=2:nmax
        dpi = dp.spec(i,:);
        dqi = dq(i,:);
        dqspeci = matrices.trialT\(dqi');
        h = h - dt*dpi*M*dqspeci;
    end
    dqend =  matrices.trialT\dq(nmax+1,:)';
    h = h -0.5*dt*dp.spec(nmax+1,:)*M*dqend;
end

%% %%%%%%%%%%%%%%%% Misc %%%%%%%%%%%%%%%%
function GradF = ComputeOptCondGradient(q,dq,y,p,...
                        L2NormInTimeP,ActiveSet,gamma,...
                        solveTangent,solveDFH,args)
    nmax=args.nmax;
    N=args.N;
    dt=args.dt;
    dq = reshape(dq,nmax+1,N+1);
    dy = solveTangent(q,y,dq,args);
    dp = solveDFH(q,y,p,dq,dy,args);

    grad1 = repmat(max(0,1-args.alpha./L2NormInTimeP),nmax+1,1).*...
        (dp.spatial);
    grad2 = args.alpha*ActiveSet.*...
        repmat(dt*sum((p.spatial).*(dp.spatial))./(L2NormInTimeP.^3),...
        nmax+1,1).*(p.spatial);
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