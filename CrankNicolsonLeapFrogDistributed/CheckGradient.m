function CheckGradient(q, dq, solveState, solveAdjoint, computeJ, computeJp, args)

u = solveState(q, args);
z = solveAdjoint(q, u, args);
jprime = computeJp(dq, q, u, z, args);

for i = 1:16
    epsilon = sqrt(10)^(-i);
    
    qp = q + epsilon*dq;
    up = solveState(qp, args);
    jp = computeJ(qp, up, args);

    qm = q - epsilon*dq; 
    um = solveState(qm, args);
    jm = computeJ(qm, um, args);
    
    %jprime = g'*dq;
    jdiff = 0.5*(jp - jm) / epsilon;
    rerr(i) = abs(jprime - jdiff) / abs(jprime);
    fprintf('jp: %f, difference quot.: %f, rel. err.: %e\n', jprime, jdiff, rerr(i));
end

semilogy(rerr);

end
