function CheckHessian(q, dq, solveState, solveAdjoint, solveTangent, solveDFH, ...
    computeJ, computeJpp, args)

    u = solveState(q, args);
    j = computeJ(q, u, args);
    z = solveAdjoint(q, u, args);
    du = solveTangent(q, u, dq, args);
    dz = solveDFH(q, u, z, dq, du, args);
    jpp = computeJpp(q, u, z, dq, du, dz, args);

    for i = 1:10
        epsilon = sqrt(10)^(-i);

        qp = q + epsilon*dq;
        up = solveState(qp, args);
        jp = computeJ(qp, up, args);

        qm = q - epsilon*dq; 
        um = solveState(qm, args);
        jm = computeJ(qm, um, args);

        %jpp = jppdq'*dq;
        jdiff = (jp - 2*j + jm) / epsilon^2;
        rerr(i) = abs(jpp - jdiff) / abs(jpp);
        fprintf('jpp: %f, difference quot: %f, rel. err.: %e\n',...
            jpp, jdiff, rerr(i));
    end
    semilogy(rerr);
end
