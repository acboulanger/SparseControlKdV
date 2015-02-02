function CheckHessian(u, du, solveState, solveAdjoint, solveTangent, solveDFH, ...
    compute_j, compute_second_derivatives_j, args)

    y = solveState(u, args);
    j = compute_j(u, y, args);
    z = solveAdjoint(u, y, args);
    dy = solveTangent(u, y, du, args);
    dz = solveDFH(u, y, z, du, dy, args);
    h = compute_second_derivatives_j(u, y.spatial, z.spatial, ...
        du, dy.spatial, dz.spatial, args);
    du2 = du(1:end-1,:);% last step does not count - adjoint has nmax +1 steps while state has nmax+2
    jpp = h'*args.matrices.Mass*du2(:);

    for i = 1:10
        epsilon = sqrt(10)^(-i);

        up = u + epsilon*du;
        yp = solveState(up, args);
        jp = compute_j(up, yp, args);

        um = u - epsilon*du; 
        ym = solveState(um, args);
        jm = compute_j(um, ym, args);

        %jpp = jppdq'*dq;
        jdiff = (jp - 2*j + jm) / epsilon^2;
        rerr(i) = abs(jpp - jdiff) / abs(jpp);
        fprintf('jpp: %f, difference quot: %f, rel. err.: %e\n',...
            jpp, jdiff, rerr(i));
    end
    semilogy(rerr);
end
