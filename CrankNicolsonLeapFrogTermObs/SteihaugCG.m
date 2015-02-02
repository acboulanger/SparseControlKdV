function [x, flag, relres, iter] = SteihaugCG(H, b, rtol, maxit, A, sigma, DP)
    % A is the mass matrix
    % H is a vector containing A
    % b is a vector containing A

    n = size(b, 1);
    x = zeros(n, 1);

    % inner product
    in = @(Av, w) Av' * DP(w);
    full_in = @(Av, w) Av' * w;

    r = b;
    z = A \ r;
    d = z;
    delta = in(r, z);

    res0 = sqrt(delta);
    iter = 0;
    flag = 'converged';

    %fprintf('iter: %i res: %f\n', iter, res0);

    while (sqrt(delta) > res0*rtol)
        iter = iter + 1;

        if(iter > maxit)
            flag = 'maxit';
            iter = maxit;
            break;
        end

        Hd = H(d);

        gamma = in(Hd, d);

        % negative curvature:
        if (gamma <= 0) 
            flag = 'neg. def.';
            relres = sqrt(delta) / res0;
            x = to_boundary(x, d, sigma, full_in, A);
            return;
        end

        alpha = delta / gamma;

        x = x + alpha * d;
        r = r - alpha * Hd;

        % trust region radius reached
        normx = sqrt(in(A*x, x));
        if (normx > sigma)
            flag = 'radius';
            relres = sqrt(delta) / res0;
            x = x - alpha * d;
            x = to_boundary(x, d, sigma, full_in, A);
            return;
        end

        deltaold = delta;

        %if (mod(iter, 50) == 0)
        %    r = b - H(x);
        %end

        z = A \ r;
        delta = in(r, z);

        beta = - alpha * in(Hd, z) / deltaold;
        %beta = delta / deltaold;

        d = z + beta * d;

        if (mod(iter, 10) == 0)
          fprintf('\t\tpcg %i: relres: %e |x|: %e\n', iter, sqrt(delta) / res0, sqrt(full_in(A*x, x)));
        end
    end

   
   % minimize |H(x + \theta z) - b|^2_Ai = |r + \theta H(z)|^2_Ai
   Hz = H(z);
   AiHz = A \ Hz;
   theta = (r' * AiHz) / (Hz' * AiHz);
   normz = sqrt(full_in(A*z, z));
   theta = min(sigma/normz,theta);
   x = x - theta * z;
   fprintf('\t\ttheta = %f\n', theta);
   

   r = b - H(x);
   relres = sqrt(r' * (A \ r)) / res0;

end

function x = to_boundary(x, d, s, in, A)

    dd = in(A*d, d);
    xd = in(A*x, d);
    xx = in(A*x, x);

    det = xd*xd + dd * (s*s - xx);
    tau = (s*s - xx) / (xd + sqrt(det));

    x = x + tau * d;

end