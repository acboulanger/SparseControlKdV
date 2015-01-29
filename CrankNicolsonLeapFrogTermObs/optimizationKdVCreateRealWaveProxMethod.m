function [u,y,args] = optimizationKdVCreateRealWaveProxMethod()
    
    ClearClose();

    args = CreateParameters();
    args.matrices = BuildMatrices(args);
    
    % observation domain
    args.matrices.Obs = ...
        ComputeObservationMatrix(1,args.N+1,args);
    args.matrices.Adjoint = args.matrices.trial*...
        args.matrices.Obs*(args.matrices.trialTInv)';
    
    % control domain
    [chi, chiT] = ...
        ComputeControlMatrix(1,args.N+1,args);
    
    % convolution of our dirac with smooth function
     [Conv,ConvT] = ComputeConvolutionMatrix(@fconvolution,args);
%     
     args.matrices.B = chi;
     args.matrices.BT = chiT;
     
     
     
     %% Check forward problem
%     u = zeros(args.nmax+1,args.N+1);%initialization of the control
%     u(:,args.N/2-5) = +2.0;
%     u(:,args.N/2) = -2.0;
%     y = solveState(u,args);% one forward simulation for y
%     plottedsteps=1:2:size(y.spatial,1);
%     [tg,xg] = meshgrid(args.tdata(plottedsteps),args.chebyGL(1:end));
%     
%     surf(xg,tg,y.spatial(plottedsteps,:)');
%     xlabel('x');ylabel('Time');zlabel('State variable y');
%     title('State Variable y');
%     view(-16,10);
%     shading interp;
    

%% Uncomment if goal is: create a specific wave at final time
    args.kappa = 1.0;
    args.x0 = 2.0;
    args.yobs =12*args.kappa^2*sech(args.kappa*(args.chebyGL - args.x0)).^2;%valeurs aux chebypoints
    args.yspecobs = args.matrices.trialT\(args.yobs)';
    
%%  Start of Trust Region Steihaug globalization strategy
    fprintf('Steihaug-CG globalization strategy...\n');
    delta = args.delta;
    gamma = args.gamma;% regularization term in 1/gamma
    fprintf('gamma = %d , delta = %d\n', gamma, delta);
    abstol=1e-6;
    %     
    q = 0.1*ones((args.nmax+1)*(args.N+1),1);%initialization of the normal variable
    qold = q;
% 
    fold = inf;
    sigma = args.sigma;% Trust region radius
    sigmamax = args.sigmamax;
    
    for iter=1:(args.maxiter)
        % compute the current iterate for the control
        % u = P_gamma(q)
        % where P_gamma is the proximal operator
        u = proximalOp(q,args);
        % solve state equation y=S(u)
        y = solveState(u,args);% one forward simulation for y
        
        % compute reduced objective functional
        % f(u) = j + L12 + L22
        % f(u) = 1/2|y(u) - yd|^2 + gamma/2*|u|_{L2(L2)}^2 + alpha|u|_{L1(L2)}
        j = compute_j(u,y,args);
        norm12 = args.alpha*sum(args.matrices.MassS*( sqrt(args.dt*sum(u.*u)) )');
        uvec = u(:);
        norm22 = 0.5*gamma*uvec'*args.matrices.Mass*uvec;
        f = j + norm12 + norm22;
        
        if iter > 1
            %check for descent in objective functional
            sigmaold = sigma;
            if(fold < (1-1e-11)*f)%not a descent direction
                sigma = 0.2*sigma;
                fprintf('\t Reject step since %1.9e >~ %1.9e \t %1.2e -> %1.2e \n',...
                    f, fold, sigmaold, sigma);
                q = qold;
                u = uold;
                y = yold;
            else
                %evaluate the decrease predicted
                %by the quadratic model (at the old iterate q)
                %m(dv)
                du = proximalOpDerivative(q,dq,args);
                DGdq = compute_reduced_hessian(dq,u,y,p,args);%TO DO, result shall be vector
                model = G'*du(:) + ...
                    0.5*DGdq'*du(:);
                rho = (f - fold)/model;
                
                if(abs(rho-1) < 0.2)%trust region might be too small
                    sigma = max(2*sigma, sigmamax);
                elseif(abs(rho-1) > 0.6)%trust region is be too big, no good approximation
                    sigma = 0.4*sigma;
                end
                fprintf('\t rho = %f, \t %1.2e -> %1.2e \n', ...
                    rho, sigmaold, sigma);
            end
        end  
        qold = q;
        uold = u;
        yold = y;
        fold = f;

        % reduced subgradient
        %
        % G(q) = q + \nabla j(P_gamma(a))
        % NB: we want G(q) = 0
        p = solveAdjoint(u,y,args);
        dj = compute_derivatives_j(u,y.spatial,p.spatial,args);
        G = args.matrices.Mass*(dj + args.gamma*q);
        
        % stopping criterion
        G2 = args.matrices.Mass\G;
        res = sqrt(G'*(args.matrices.Mass\G));
        fprintf('%d: f = %e, res=%e\n', iter,f,res);
        if(res < abstol)
            break
        end
        % compute the Newton update
        % DG(q) dq = -G(q)
        DG = @(dq) compute_reduced_hessian(dq, u, y, p, args);

        % with M(odified)PCG
        DP = @(dq) proximalOpDerivative(q,dq,args);
        [dq, flag, relres, pcggit] = SteihaugCG(DG, -G, 1e-5, floor(args.nmax/3),...
            args.matrices.Mass, sigma, DP);
        fprintf('krylov %s: iter=%d, relres=%e, |dq|=%e\n', ...
            flag, pcggit, relres, sqrt(dq'*args.matrices.Mass*dq))

        % apply (TR)-Newton update
        q = qold + dq;
        myvisu(y.spatial,p.spatial,u,gamma,args);
    end
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
    args.D = 30; %domain is -50..50
    args.N = 256; %number of points
    args.k = args.N:-1:0;

    %Creation of Chebyshev Gauss-Lobatto points - our nodal basis
    args.chebyGL = cos(args.k*pi/args.N)*args.D;
    args.npoints = size(args.chebyGL,2);
    args.spacestep = [(args.chebyGL(2:end) - args.chebyGL(1:end-1))] ;
    args.ncells = args.npoints-1;

    %time argseters
    args.dt = 0.01;% time step for simulation
    args.tmax = 5.00;% maximum time for simulation
    args.nmax = round(args.tmax/args.dt);% induced number of time steps
    args.tdata = args.dt*(0:1:(args.nmax+1));
    args.maxiter = 1e3;

    % Optimization parameters
    args.alpha = 0.0001;
    args.iterNewton = 5;
    args.tolNewton = 1e-5;
    args.epsilon = 1e-12;
    args.tolgmres = 1e-3;

    % Trust region Steihaug globalization
    args.gammaArray = 2.^[4:20];
    args.gamma = 1e5;
    args.delta = 1.0;
    args.sigma = 10;
    args.sigmamax = 100;
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
    args.yobs = zeros(1,args.N+1);
    args.yspecobs = zeros(1,args.N-2)';
    args.q = 0.0*ones(args.nmax+2, args.N+1);
    
    args.normp = zeros(1,args.N+1);
    
    % physical parameters
    args.f = 0.750;
    args.coeff3d = -1.0/6.0;
    args.coeffburgers = 3.0/2.0;
    args.coeffsource = 1.0/2.0;

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
    P = -spdiags([pdiagm2' pdiagm1' pdiag0' pdiag1' pdiag2'],...
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
    matrices.Msource = 0.5*M;
    matrices.MT = M';
    eps = 2*args.D/(args.N);
    matrices.Mreg = matrices.M + eps*eye(size(matrices.M,1));
    matrices.MTreg = matrices.MT + eps*eye(size(matrices.MT,1));
    matrices.MTInv = inv(M');
    matrices.S = args.coeff3d*S;
    matrices.P = P;
    matrices.PT = P';
    matrices.Pnl = args.coeffburgers*P;
    matrices.PnlT = (matrices.Pnl)';
    matrices.fP = args.f*P;
    matrices.fPT = (matrices.fP)';
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
    
    % control is discretized with linear finite elements in space
    % piecewise constant in time
    % construct the lumped mass matrix for space
    dx = [0,args.spacestep,0];
    matrices.MassS = spdiags(0.5*(dx(2:end) + dx(1:(end-1)))',...
        0,args.N+1, args.N+1);
    matrices.MassSSource = 0.5*matrices.MassS;
    % construct the lumped mass matrix for time (actually P0 in time)
    matrices.MassT = args.dt*speye(args.nmax+1);
    % construct the matrix A for the inner product in H (space and time)
    % (u,v)_H = u'Av
    matrices.Mass = kron(matrices.MassS,matrices.MassT);%should be diag(MassT(i)*MassS)
    
end

function [Obs] = ComputeObservationMatrix(i1,i2,args)
    observationdomain = i1:i2;
    Obs = zeros(args.N+1);
    for i=1:size(observationdomain,2)
        Obs(observationdomain(i), observationdomain(i)) = 1;
    end
end

function [B,BT] = ComputeControlMatrix(i1,i2,args)
    controldomain = i1:i2;
    B = zeros(args.N+1);
    for i=1:size(controldomain,2)
        B(controldomain(i), controldomain(i)) = 1.0;
    end
BT = B';  
end

function res = fconvolution(x,x_center)
    %res = -0.626657*exp(-0.5*(x-x_center).^2).*(x-x_center);
    res = (x==x_center);
end

function [Conv,ConvT] = ComputeConvolutionMatrix(fconv,args)
    Conv = zeros(args.N+1, args.N+1);
    for i=1:args.N+1
        xi = args.chebyGL(i);
        for j=1:(args.N+1)
            xj = args.chebyGL(j);
            Conv(i,j) = fconv(xi,xj);
        end
    end
    ConvT = Conv';
end

%% %%%%%%%%%%%%%%%% solveState functions %%%%%%%%%%%%%%%%
function y = solveState(u, args)
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
    u = ((args.matrices.B)*(u'))';%effect of indicator function
    
    %uspec = matrices.trialT\u';
    %uspec=uspec';

    % first time step in the spectral space, semi implicit
    NLterm = (args.y0).^2;
    pNLterm=coeffNL*matrices.trialTInv*NLterm';
    yspec1 = matrices.leftInv*((0.5*matrices.M)*yspec0 ...
            + 0.5*dt*(-0.5*matrices.Pnl*pNLterm + matrices.fP*yspec0...
            + matrices.trial*matrices.MassSSource*u(1,:)'));
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
          + dt*(-0.5*matrices.Pnl*pNLterm...
          + matrices.fP*yspecm1 ...
          + matrices.trial*matrices.MassSSource*u(i,:)') );
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
    
%     rhs = ((matrices.right)*yspecm2 +...
%                 0.5*dt*(0.5*matrices.P*pNLterm + matrices.P*yspecm1)...
%                 + 0.5*matrices.M*yspecm1...
%                 + 1.0*0.5*dt*matrices.M*qspec(nmax+1,:)');
%     [yspecend,success,residual,itermeth] = gmres(matrices.M,rhs,[],args.tolgmres,N-2);
    yspecend = (matrices.Mreg)\((matrices.right)*yspecm2 +...
                0.5*dt*(-0.5*matrices.Pnl*pNLterm + matrices.fP*yspecm1)...
                + 0.5*matrices.M*yspecm1...
                + 1.0*0.5*dt*matrices.trial*matrices.MassSSource*u(nmax+1,:)');
    yend = matrices.trialT*yspecend;
    y.spec(end,:) = yspecend;
    y.spatial(end,:) = yend;
end


%% %%%%%%%%%%%%%%%% solveAdjoint functions %%%%%%%%%%%%%%%%
function p = solveAdjoint(u,y,args)
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

    % initialization 
    % M is ill conditionned. Regularization added
    mt = matrices.MTreg;
    %rhs = (y.spec(end,:)' - args.yspecobs);
    %[pspec0,success,residual,itermeth] = gmres(mt,rhs,[],args.tolgmres,N-2);
    rhsspatial = args.matrices.Obs*(y.spatial(end,:)'- args.yobs');
    rhs = matrices.trialT\rhsspatial;
    %pspec0 = -mt\(matrices.A*(y.spec(end,:)' - args.yspecobs));
    pspec0 = -mt\(matrices.Adjoint*matrices.A*rhs);
    p0 = (matrices.testT)*(pspec0);
   
    p.spatial(1,:) = p0;
    p.spec(1,:) = pspec0;

    % first step
    NLterm = 0.5*2.0*yrev(2,:)'.*...
        (matrices.trialTInv'*(matrices.PnlT*pspec0));
    pNLterm=coeffNL*matrices.trial*NLterm;
    pspec1 = matrices.leftTInv*(0.5*matrices.MT*pspec0 +...
        0.5*dt*(matrices.fPT*pspec0 - pNLterm));
    p1 = matrices.testT*pspec1;
    p.spatial(2,:) = p1;
    p.spec(2,:) = pspec1;
    
    pspecm1 = pspec1;
    pspecm2 = pspec0;
    %Time loop
    for i = 2:nmax
        NLterm = 0.5*2.0*yrev(i+1,:)'.*...
            (matrices.trialTInv'*(matrices.PnlT*pspecm1));
        pNLterm=coeffNL*matrices.trial*NLterm;
        pspeci = matrices.M_leftTinv_rightT*pspecm2...
            + matrices.M_leftTinv_dt* (matrices.fPT*pspecm1 - pNLterm);
        pi = matrices.testT*pspeci;
        pspecm2 = pspecm1;
        pspecm1 = pspeci; 
        p.spec(i+1,:) = pspeci;
        p.spatial(i+1,:) = pi;
    end
    p.spec = p.spec(end:-1:1,:);
    p.spatial = p.spatial(end:-1:1,:);
end

%% %%%%%%%%%%%%%%%% Tracking term %%%%%%%%%%%%%%%%
function j = compute_j(u,y,args)
    discr = args.matrices.Obs*(y.spatial(end,:)'-args.yobs');
    ymyd = args.matrices.trialT\discr;
    j = 0.5*ymyd'*(args.matrices.A*ymyd);
end

function dj = compute_derivatives_j(u,y,p,args)%do not forget time in inner product
    %p: row = time, column = space
    dj = (args.matrices.BT)*(p)';%each column is B*p(t_i)
    dj = dj(:);%makes a vector
end

function ddj = compute_second_derivatives_j(u, y, p, du, dy, dp, args)
    ddj = (args.matrices.BT)*(dp)';%each column is B*dp(t_i)
    ddj = ddj(:);%makes a vector
end

%% %%%%%%%%%%%%%%%% solveTangent functions %%%%%%%%%%%%%%%%
function dy = solveTangent(u, y, du, args)
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
    du = (args.matrices.B*(du'))';%effect of indicator function
    %duspec = matrices.trialTInv*du';
    %duspec=duspec';
    dy.spatial(1,:) = args.dy0;
    dy.spec(1,:) = dyspec0;

    %first time step in the spectral space, semi implicit
    NLterm = y.spatial(1,:).*(args.dy0);
    pNLterm = coeffNL*matrices.trialTInv*NLterm';
    dyspec1 = matrices.leftInv*((0.5*matrices.M)*dyspec0 ...
        + 0.5*dt*(-matrices.Pnl*pNLterm + matrices.fP*dyspec0 ...
        + matrices.trial*matrices.MassSSource*du(1,:)'));
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
          + dt*(-matrices.Pnl*pNLterm...
          + matrices.fP*dyspecm1 ...
          + matrices.trial*matrices.MassS*du(i,:)') );
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
    
%     rhs = ((matrices.right)*dyspecm2...
%                     + 0.5*dt*(matrices.P*pNLterm + matrices.P*dyspecm1)...
%                     + 0.5*matrices.M*dyspecm1...
%                     + 1.0*0.5*dt*matrices.M*dqspec(nmax+1,:)');
%     [dyspecend,success,residual,itermeth] = gmres(matrices.M,rhs,[],args.tolgmres,N-2);
    dyspecend = (matrices.Mreg)\((matrices.right)*dyspecm2...
                   + 0.5*dt*(-matrices.Pnl*pNLterm + matrices.fP*dyspecm1)...
                   + 0.5*matrices.M*dyspecm1...
                   + 1.0*0.5*dt*matrices.trial*matrices.MassSSource*du(nmax+1,:)');
     dyend = matrices.trialT*dyspecend;
     dy.spec(end,:) = dyspecend;
     dy.spatial(end,:) = dyend;
end


%% %%%%%%%%%%%%%%%% solveDFH functions %%%%%%%%%%%%%%%%
function dp = solveDFH(u, y, p, du, dy, args)
    dt=args.dt;
    nmax=args.nmax;
    N=args.N;
    coeffNL = args.coeffNL;
    matrices = args.matrices;

    %source term
    yrev = y.spatial(end:-1:1,:);
    dyrev = dy.spatial(end:-1:1,:);
    pspecrev=p.spec(end:-1:1,:);

    % Storage array
    dp.spatial = zeros(nmax+1,N+1);
    dp.spec = zeros(nmax+1,N-2);

    %initial condition
    mt = matrices.MTreg;
    %rhs = -matrices.A*(dy.spec(end,:)');
    %[dpspec0,success,residual,itermeth] = gmres(mt,rhs,[],args.tolgmres,N-2);
    rhsspatial = args.matrices.Obs*(dy.spatial(end,:)');
    rhsspec = matrices.trialT\rhsspatial;
    dpspec0= -mt\(matrices.Adjoint*matrices.A*(rhsspec));
    dp0 = (matrices.testT)*(dpspec0);
    
    dp.spatial(1,:) = dp0;
    dp.spec(1,:) = dpspec0;

    %first step

    NLterm = yrev(2,:)'.*(matrices.trialTInv'*(matrices.PnlT*dpspec0));
    NLterm2 = dyrev(2,:)'.*...
        (matrices.trialTInv'*(matrices.PnlT*pspecrev(1,:)'));
    pNLterm = coeffNL*matrices.trial*(NLterm+NLterm2);

    dpspec1 = matrices.leftTInv*(0.5*matrices.MT*dpspec0 + ...
        0.5*dt*(matrices.fPT*dpspec0 - pNLterm));
    dp1 = matrices.testT*dpspec1;

    dp.spatial(2,:) = dp1;
    dp.spec(2,:) = dpspec1;
    dpspec2 = dpspec0;
    
    % Time loop
    for i = 2:(nmax)
        NLterm = yrev(i+1,:)'.*(matrices.trialTInv'*(matrices.PnlT*dpspec1));
        NLterm2 = dyrev(i+1,:)'.*...
            (matrices.trialTInv'*(matrices.PnlT*pspecrev(i,:)'));
        pNLterm = coeffNL*matrices.trial*(NLterm+NLterm2);

        dpspeci=matrices.M_leftTinv_rightT*dpspec2...
                + matrices.M_leftTinv_dt* (matrices.fPT*dpspec1 - pNLterm );
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


%% %%%%%%%%%%%%%%%% Proximal map %%%%%%%%%%%%%%%%
function u = proximalOp(q,args)
    %q is here a vector - needs to be transformed in a matrix
    gamma = args.gamma;
    q = reshape(q,args.nmax+1,args.N+1);
    L2NormInTimeQ = sqrt(sum(args.matrices.MassT*((q).*...
        (q))));
    %size(L2NormInTimeQ)
    for k=1:(args.N+1)    % check norm 0           
        if (L2NormInTimeQ(k) <= args.epsilon)
            L2NormInTimeQ(k) = gamma*(args.alpha - args.epsilon);
        end
    end
    u = 1/gamma*repmat(max(0,gamma-args.alpha./L2NormInTimeQ),...
        args.nmax+1,1).*(q);
end

function dpc = proximalOpDerivative(q,dq,args)
    % q and dq are vectors - needs to be reshaped in matrices
    gamma = args.gamma;
    nmax = args.nmax;
    MassT = args.matrices.MassT;
    q = reshape(q,args.nmax+1,args.N+1);
    dq = reshape(dq,args.nmax+1,args.N+1);
    L2NormInTimeQ = sqrt(sum(MassT*((q).*...
    (q))));
    for k=1:(args.N+1)    % check norm 0           
        if (L2NormInTimeQ(k) <= args.epsilon)
            L2NormInTimeQ(k) = gamma*(args.alpha - args.epsilon);
        end
    end
    ActiveSet = repmat(L2NormInTimeQ > args.alpha/args.gamma,args.nmax+1,1);
    dpc = 1/args.gamma*ActiveSet.*...
        (repmat(max(0,args.gamma-args.alpha./L2NormInTimeQ),nmax+1,1).*dq + ...
            repmat(sum(MassT*((q).*(dq)))./(L2NormInTimeQ.^3),nmax+1,1).*(q));
    dpc = dpc(:);
end

%% %%%%%%%%%%%%%%%%% Reduced Hessian %%%%%%%%%%%%%%%
function DGh = compute_reduced_hessian(dq,u,y,p,args)
% dq is an array
% Newton derivative of G
%
% DG(q)dq = dq + Hj(Pc(q))DPc(q)dq
%
% NB: chain rule for semismoothness
    du = proximalOp(dq,args);
    dy = solveTangent(u, y, du, args);
    dp = solveDFH(u, y, p, du, dy, args);
    ddj = compute_second_derivatives_j(u,y.spatial,p.spatial,...
        du,dy.spatial,dp.spatial,args);

    DGh = args.matrices.Mass*(args.gamma*dq + ddj);%include mass matrix in result

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% VISUALIZATION %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
    
    subplot(2,3,6), plot(args.chebyGL(1:end),args.matrices.Obs*(y(end,:)'-args.yobs'));
    xlabel('x');zlabel('Y - Yobs');
    title('Error');

    str = sprintf('Continuation strategy, gamma = %d', gamma);
    suptitle(str);
    
    drawnow();
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