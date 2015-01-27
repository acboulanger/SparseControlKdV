function h = CG(q,solveState,solveAdjoint,...
    solveTangent,solveDFH,args)

end

function h = SteihaugCG(q,solveState,solveAdjoint,...
    solveTangent,solveDFH,args)
    
    % Initialization of the residual
    u = proximalMap(q);
    y = solveState(u,args);
    p = solveAdjoint(u,y,args);
    
    r = -args.gamma*q + p;
    d = r;
    h = zeros(size(q));
    
    for k=1..args.nSCG
       Adk = ;
       if()
    end
    
    
end

function u = proximalMap(q,args)

end