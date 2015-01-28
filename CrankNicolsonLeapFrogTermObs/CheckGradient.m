function CheckGradient(q, dq, solveState, solveAdjoint, computeJ, computeJp, args)

y = solveState(u, args);
z = solveAdjoint(u, y, args);
g = compute_derivative_j(u, y, z, args);
jprime = g'*dq;

for i = 1:16
    epsilon = sqrt(10)^(-i);
    
    qp = q + epsilon*dq;
    up = solveState(qp, args);
    jp = compute_j(qp, up, args);

    qm = q - epsilon*dq; 
    um = solveState(qm, args);
    jm = compute_j(qm, um, args);
    
    %jprime = g'*dq;
    jdiff = 0.5*(jp - jm) / epsilon;
    rerr(i) = abs(jprime - jdiff) / abs(jprime);
    fprintf('jp: %f, difference quot.: %f, rel. err.: %e\n', jprime, jdiff, rerr(i));
end

semilogy(rerr);

end
