function CheckGradient(u, du, solveState, solveAdjoint, compute_j, compute_derivatives_j, args)

y = solveState(u, args);
z = solveAdjoint(u, y, args);
g = compute_derivatives_j(u, y.spatial, z.spatial, args);

dup = du(1:end-1,:);% last step does not count - adjoint has nmax +1 steps while state has nmax+2
jprime = g'*args.matrices.Mass*dup(:);

for i = 1:16
    epsilon = sqrt(10)^(-i);
    
    up = u + epsilon*du;
    yp = solveState(up, args);
    jp = compute_j(up, yp, args);

    um = u - epsilon*du; 
    ym = solveState(um, args);
    jm = compute_j(um, ym, args);
    
    %jprime = g'*dq;
    jdiff = 0.5*(jp - jm) / epsilon;
    rerr(i) = abs(jprime - jdiff) / abs(jprime);
    fprintf('jp: %f, difference quot.: %f, rel. err.: %e\n', jprime, jdiff, rerr(i));
end

semilogy(rerr);

end
